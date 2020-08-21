#!/usr/bin/env python

## @file jiminy_py/viewer.py

import os
import re
import io
import time
import psutil
import shutil
import signal
import base64
import atexit
from tqdm import tqdm
import asyncio
import umsgpack
import tempfile
import subprocess
import logging
import webbrowser
import numpy as np
import tornado.web
import multiprocessing
from PIL import Image
from bisect import bisect_right
from threading import Thread, Lock
from contextlib import redirect_stdout, redirect_stderr

import zmq
import meshcat
import meshcat.transformations as mtf
from meshcat.servers.zmqserver import (
    VIEWER_ROOT, ZMQWebSocketBridge, WebSocketHandler)

import pinocchio as pin
from pinocchio import SE3, se3ToXYZQUAT, XYZQUATToSe3
from pinocchio.rpy import rpyToMatrix, matrixToRpy
from pinocchio.robot_wrapper import RobotWrapper

from .state import State


# Determine if the various backends are available
backends_available = ['meshcat']
if __import__('platform').system() == 'Linux':
    try:
        import gepetto as _gepetto
        import omniORB as _omniORB
        backends_available.append('gepetto-gui')
    except ImportError:
        pass

# Monkey-patch subprocess Popen to add 'is_alive' and 'join' methods,
# to have the same interface than multiprocessing Process.
def is_alive(self):
    return self.poll() is None
subprocess.Popen.is_alive = is_alive
subprocess.Popen.join = subprocess.Popen.wait

CAMERA_INV_TRANSFORM_MESHCAT = rpyToMatrix(np.array([-np.pi/2, 0.0, 0.0]))
DEFAULT_CAMERA_XYZRPY = np.array([7.5, 0.0, 1.4, 1.4, 0.0, np.pi/2])


def sleep(dt):
    """
        @brief   Function to provide cross-plateform time sleep with maximum
                 accuracy.

        @details Use this method with cautious since it relies on busy looping
                 principle instead of system scheduler. As a result, it wastes
                 a lot more resources than time.sleep. However, it is the only
                 way to ensure accurate delay on a non-real-time systems such
                 as Windows 10.

        @param   dt   sleep duration in seconds.
    """
    _ = time.perf_counter() + dt
    while time.perf_counter() < _:
        pass


def start_zmq_server():
    # Monkey-patch Meshcat to support cross-origin connection.
    # It is useful to execute custom javascript commands within
    # a Jupyter Notebook, and it is not an actual security flaw
    # for local servers since they are not accessible from the
    # outside anyway.
    WebSocketHandler.check_origin = lambda self, origin: True

    # Override the default html page to disable auto-update of
    # three js "controls" of the camera, so that it can be moved
    # programmatically in any position, without any constraint, as
    # long as the user is not moving it manually using the mouse.
    class MyFileHandler(tornado.web.StaticFileHandler):
        def initialize(self, default_path, default_filename, fallback_path):
            self.default_path = os.path.abspath(default_path)
            self.default_filename = default_filename
            self.fallback_path = os.path.abspath(fallback_path)
            super().initialize(self.default_path, self.default_filename)

        def set_extra_headers(self, path):
            self.set_header('Cache-Control',
                            'no-store, no-cache, must-revalidate, max-age=0')

        def validate_absolute_path(self, root, absolute_path):
            if os.path.isdir(absolute_path):
                absolute_path = os.path.join(absolute_path, self.default_filename)
                return self.validate_absolute_path(root, absolute_path)
            if os.path.exists(absolute_path) and \
                    os.path.basename(absolute_path) != self.default_filename:
                return super().validate_absolute_path(root, absolute_path)
            return os.path.join(self.fallback_path, absolute_path[(len(root)+1):])

    def make_app(self):
        return tornado.web.Application([
            (r"/static/?(.*)", MyFileHandler, {
                "default_path": VIEWER_ROOT,
                "fallback_path": os.path.join(os.path.dirname(__file__), "meshcat"),
                "default_filename": "index.html"}),
            (r"/", WebSocketHandler, {"bridge": self})
        ])
    ZMQWebSocketBridge.make_app = make_app

    # Implement bidirectional communication because zmq and the
    # websockets by gathering and forward messages received from
    # the websockets to zmq. Note that there is currently no way
    # to identify the client associated to each reply, but it is
    # usually not a big deal, since the same answers is usual
    # expected from each of them. Comma is used as a delimiter.
    #
    # It also fixes flushing issue when 'handle_zmq' is not directly
    # responsible for sending a message through the zmq socket.
    def handle_web(self, message):
        self.bridge.websocket_messages.append(message)
        if len(self.bridge.websocket_messages) == len(self.bridge.websocket_pool):
            gathered_msg = ",".join(self.bridge.websocket_messages)
            self.bridge.zmq_socket.send(gathered_msg.encode("utf-8"))
            self.bridge.zmq_stream.flush()
    WebSocketHandler.on_message = handle_web

    def wait_for_websockets(self):
        if len(self.websocket_pool) > 0:
            self.zmq_socket.send(b"ok")
            self.zmq_stream.flush()
        else:
            self.ioloop.call_later(0.1, self.wait_for_websockets)
    ZMQWebSocketBridge.wait_for_websockets = wait_for_websockets

    handle_zmq_orig = ZMQWebSocketBridge.handle_zmq
    def handle_zmq(self, frames):
        self.websocket_messages = []  # Used to gather websocket messages
        cmd = frames[0].decode("utf-8")
        if cmd == "meshes_loaded":
            if not self.websocket_pool:
                self.zmq_socket.send("".encode("utf-8"))
            for websocket in self.websocket_pool:
                websocket.write_message(umsgpack.packb({
                    u"type": u"meshes_loaded"
                }), binary=True)
        else:
            handle_zmq_orig(self, frames)
    ZMQWebSocketBridge.handle_zmq = handle_zmq

    # Meshcat server deamon, using in/out argument to get the
    # zmq url instead of reading stdout as it was.
    def meshcat_zmqserver(zmq_url):
        # Do NOT use the original even loop, if any, to avoid
        # Runtime Error in Jupyter "event loop is already running".
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with open(os.devnull, 'w') as f:
            with redirect_stderr(f):
                bridge = ZMQWebSocketBridge()
                info['zmq_url'] = bridge.zmq_url
                info['web_url'] = bridge.web_url
                bridge.run()

    # Run meshcat server in background using multiprocessing
    # Process to enable monkey patching and proper interprocess
    # communication through a manager.
    manager = multiprocessing.Manager()
    info = manager.dict()
    server = multiprocessing.Process(target=meshcat_zmqserver, args=(info,))
    server.daemon = True
    server.start()

    # Wait for the process to finish initialization
    while not info:
        pass
    return server, info['zmq_url'], info['web_url']

def start_zmq_server_standalone():
    import argparse
    argparse.ArgumentParser(
        description="Serve the Jiminy MeshCat HTML files and listen for ZeroMQ commands")

    server, zmq_url, web_url = start_zmq_server()
    print(zmq_url)
    print(web_url)

    try:
        server.join()
    except KeyboardInterrupt:
        server.terminate()
        server.join(timeout=0.5)
        try:
            proc = psutil.Process(server.pid)
            proc.send_signal(signal.SIGKILL)
        except psutil.NoSuchProcess:
            pass


class Viewer:
    backend = None
    port_forwarding = None
    _backend_obj = None
    _backend_exceptions = ()
    _backend_proc = None
    _backend_robot_names = set()
    _lock = Lock() # Unique threading.Lock for every simulations (in the same thread ONLY!)

    def __init__(self,
                 robot,
                 use_theoretical_model=False,
                 mesh_root_path=None,
                 urdf_rgba=None,
                 lock=None,
                 backend=None,
                 delete_robot_on_close=False,
                 close_backend_at_exit=True,
                 robot_name=None,
                 window_name='jiminy',
                 scene_name='world'):
        """
        @brief Constructor.

        @param robot          The jiminy.Robot to display.
        @param use_theoretical_model   Whether to use the theoretical (rigid) model or the flexible
                                       model for this robot.
        @param mesh_root_path    Path to the folder containing the URDF meshes.
                                 Optional: Must only be specified for relative mesh paths.
                                           It will override any absolute mesh path if specified.
        @param urdf_rgba      RGBA color to use to display this robot, as a list of 4 floating-point
                              values between 0.0 and 1.0.
                              Optional: It will override the original color of the meshes if specified.
        @param lock           Custom threading.Lock
                              Optional: Only required for parallel rendering using multiprocessing.
                                        It is required since some backends does not support multiple
                                        simultaneous connections (e.g. corbasever).
        @param backend        The name of the desired backend to use for rendering. It can be
                              either 'gepetto-gui' or 'meshcat' ('panda3d' available soon).
                              Optional: 'gepetto-gui' by default if available and not running
                                        inside a notebook, 'meshcat' otherwise.
        @param delete_robot_on_close     Enable automatic deletion of the robot when closing.
        @param close_backend_at_exit     Terminate backend server at Python exit.
        @param robot_name     Unique robot name, to identify each robot in the viewer.
                              Optional: Randomly generated identifier by default.
        @param window_name    Window name, used only when gepetto-gui is used as backend.
                              Note that it is not allowed to be equal to the window name.
        @param scene_name     Scene name, used only when gepetto-gui is used as backend.
        """
        # Handling of default arguments
        if robot_name is None:
            uniq_id = next(tempfile._get_candidate_names())
            robot_name="_".join(("robot", uniq_id))

        # Backup some user arguments
        self.urdf_path = robot.urdf_path
        self.robot_name = robot_name
        self.scene_name = scene_name
        self.window_name = window_name
        self.use_theoretical_model = use_theoretical_model
        self._lock = lock if lock is not None else Viewer._lock
        self.delete_robot_on_close = delete_robot_on_close
        self.close_backend_at_exit = close_backend_at_exit

        # Make sure that the windows, scene and robot names are valid
        if scene_name == window_name:
            raise ValueError(
                "Please, choose a different name for the scene and the window.")

        if robot_name in Viewer._backend_robot_names:
            raise ValueError(
                "Robot name already exists but must be unique. Please choose a "\
                "different one, or close the associated viewer.")

        # Extract the right Pinocchio model
        if self.use_theoretical_model:
            self.pinocchio_model = robot.pinocchio_model_th
            self.pinocchio_data = robot.pinocchio_data_th
        else:
            self.pinocchio_model = robot.pinocchio_model
            self.pinocchio_data = robot.pinocchio_data

        # Select the desired backend
        if backend is None:
            if Viewer.backend is None:
                if Viewer._is_notebook() or not 'gepetto-gui' in backends_available:
                    backend = 'meshcat'
                else:
                    backend = 'gepetto-gui'
            else:
                backend = Viewer.backend
        else:
            if not backend in backends_available:
                raise ValueError("%s backend not available." % backend)

        # Update the backend currently running, if any
        if Viewer.backend != backend and Viewer._backend_obj is not None:
            Viewer.close()
            print("Different backend already running. Closing it...")
        Viewer.backend = backend

        # Configure exception handling
        if Viewer.backend == 'gepetto-gui':
            import omniORB
            Viewer._backend_exceptions = \
                (omniORB.CORBA.COMM_FAILURE, omniORB.CORBA.TRANSIENT)
        else:
            Viewer._backend_exceptions = (zmq.error.Again, zmq.error.ZMQError)

        # Check if the backend is still available, if any
        if Viewer._backend_obj is not None:
            is_backend_running = True
            if Viewer._backend_proc is not None and \
                    not Viewer._backend_proc.is_alive():
                is_backend_running = False
            if Viewer.backend == 'gepetto-gui':
                try:
                    Viewer._backend_obj.gui.refresh()
                except Viewer._backend_exceptions:
                    is_backend_running = False
            else:
                try:
                    zmq_socket = Viewer._backend_obj.gui.window.zmq_socket
                    zmq_socket.send(b"url")
                    zmq_socket.RCVTIMEO = 50
                    zmq_socket.recv()
                    zmq_socket.RCVTIMEO = -1  # -1 for limit, milliseconds otherwise
                except Viewer._backend_exceptions:
                    is_backend_running = False
            if not is_backend_running:
                Viewer._backend_obj = None
                Viewer._backend_proc = None
                Viewer._backend_exception = None
        else:
            is_backend_running = False

        # Access the current backend or create one if none is available
        self.is_backend_parent = False
        try:
            if Viewer.backend == 'gepetto-gui':
                if Viewer._backend_obj is None:
                    Viewer._backend_obj, Viewer._backend_proc = \
                        Viewer._get_client(True, self.close_backend_at_exit)
                    self.is_backend_parent = Viewer._backend_proc is not None
                self._client = Viewer._backend_obj.gui

                if not scene_name in self._client.getSceneList():
                    self._client.createSceneWithFloor(scene_name)
                if not window_name in self._client.getWindowList():
                    self._window_id = self._client.createWindow(window_name)
                    self._client.addSceneToWindow(scene_name, self._window_id)
                    self._client.createGroup(scene_name + '/' + scene_name)
                    self._client.addLandmark(scene_name + '/' + scene_name, 0.1)
                else:
                    self._window_id = int(np.where([name == window_name
                        for name in self._client.getWindowList()])[0][0])
            else:
                from pinocchio.visualize import MeshcatVisualizer
                from pinocchio.shortcuts import createDatas

                if Viewer._backend_obj is None:
                    Viewer._backend_obj, Viewer._backend_proc = \
                        Viewer._get_client(True, self.close_backend_at_exit)
                    self.is_backend_parent = Viewer._backend_proc is not None

                if self.is_backend_parent:
                    self.open_gui()

                self._client = MeshcatVisualizer(self.pinocchio_model, None, None)
                self._client.viewer = Viewer._backend_obj.gui
        except Exception as e:
            raise RuntimeError("Impossible to create or connect to backend.") from e

        # Set the default camera pose if the viewer is not running before
        if self.is_backend_parent:
            self.set_camera_transform(
                translation=DEFAULT_CAMERA_XYZRPY[:3],
                rotation=DEFAULT_CAMERA_XYZRPY[3:])

        # Backup the backend subprocess used for instantiate the robot
        self._backend_proc = Viewer._backend_proc

        # Create a unique temporary directory, specific to this viewer instance
        self._tempdir = tempfile.mkdtemp(
            prefix= "_".join([window_name, scene_name, robot_name, ""]))

        # Check for conflict in mesh path specification
        if mesh_root_path != None:
            self.urdf_path = Viewer._urdf_fix_mesh_path(
                self.urdf_path, mesh_root_path, self._tempdir)

        # Create a RobotWrapper
        if mesh_root_path is not None:
            root_path = mesh_root_path
        else:
            root_path = os.environ.get('JIMINY_MESH_PATH', [])
        if Viewer.backend == 'gepetto-gui':
            Viewer._delete_nodes_viewer([scene_name + '/' + self.robot_name])
            if urdf_rgba is not None:
                alpha = urdf_rgba[3]
                self.urdf_path = Viewer._get_colorized_urdf(
                    self.urdf_path, urdf_rgba[:3], root_path, self._tempdir)
            else:
                alpha = 1.0
        collision_model = pin.buildGeomFromUrdf(
            self.pinocchio_model, self.urdf_path,
            root_path, pin.GeometryType.COLLISION)
        visual_model = pin.buildGeomFromUrdf(
            self.pinocchio_model, self.urdf_path,
            root_path, pin.GeometryType.VISUAL)
        self._rb = RobotWrapper(model=self.pinocchio_model,
                                collision_model=collision_model,
                                visual_model=visual_model)
        if not self.use_theoretical_model:
            self._rb.data = robot.pinocchio_data
        self.pinocchio_data = self._rb.data

        # Load robot in the backend viewer
        if Viewer.backend == 'gepetto-gui':
            self._rb.initViewer(
                windowName=window_name, sceneName=scene_name, loadModel=False)
            self._rb.loadViewerModel(self.robot_name)
            self._client.setFloatProperty(scene_name + '/' + self.robot_name,
                                          'Transparency', 1 - alpha)
        else:
            self._client.collision_model = collision_model
            self._client.visual_model = visual_model
            (self._client.data,
             self._client.collision_data,
             self._client.visual_data) = createDatas(
                 self.pinocchio_model, collision_model, visual_model)
            self._client.loadViewerModel(
                rootNodeName=self.robot_name, color=urdf_rgba)
            self._rb.viz = self._client
            Viewer._backend_obj.info['nmeshes'] += \
                len(self._rb.visual_model.geometryObjects)
        Viewer._backend_robot_names.add(robot_name)

        # Refresh the viewer since the position of the meshes is not initialized at this point
        self.refresh()

    def __del__(self):
        """
        @brief Destructor.

        @remark It automatically close the viewer before being garbage collected.
        """
        self.close()

    @staticmethod
    def reset_port_forwarding(port_forwarding=None):
        """
        @brief Configure port forwarding. Only used for remote display in Jupyter notebook cell.

        @param port_forwarding  Dictionary whose keys are ports on local machine,
                                and values are associated remote port.
        """
        Viewer.port_forwarding = port_forwarding

    @staticmethod
    def _get_client_url():
        if Viewer.backend == 'gepetto-gui':
            raise RuntimeError("Can only get client url using Meshcat backend.")
        if Viewer._backend_obj is None:
            raise RuntimeError("Viewer not connected to any running Meshcat server.")

        viewer_url = Viewer._backend_obj.gui.url()
        if Viewer.port_forwarding is not None:
            url_port_pattern = '(?<=:)[0-9]+(?=/)'
            port_localhost = int(re.search(url_port_pattern, viewer_url).group())
            if not port_localhost in Viewer.port_forwarding.keys():
                raise RuntimeError("Port forwarding defined but no port "\
                                    "mapping associated with {port_localhost}.")
            port_remote = Viewer.port_forwarding[port_localhost]
            viewer_url = re.sub(url_port_pattern, str(port_remote), viewer_url)
        return viewer_url

    @staticmethod
    def open_gui(start_if_needed=False):
        """
        @brief Open a new viewer graphical interface.

        @remark  This method is not supported by Gepetto-gui since it does not have a classical
                 server-client mechanism. One and only one graphical interface (client) can be
                 opened, and its lifetime is tied to the one of the server itself.

        @param port_forwarding  Dictionary whose keys are ports on local machine,
                                and values are associated remote port.
        """
        if Viewer.backend == 'gepetto-gui':
            raise RuntimeError(
                "Showing client is only available using 'meshcat' backend.")
        else:
            if Viewer._backend_obj is None:
                if start_if_needed:
                    Viewer._backend_obj, Viewer._backend_proc = \
                        Viewer._get_client(True)
                else:
                    raise RuntimeError("No meshcat backend available and "\
                                       "'start_if_needed' is set to False.")
            if Viewer._is_notebook() and Viewer.port_forwarding is not None:
                logging.warning(
                    "Impossible to open web browser programmatically for Meshcat "\
                    "through port forwarding. Either use Jupyter or open it manually.")

            viewer_url = Viewer._get_client_url()
            if Viewer._is_notebook():
                from IPython.core.display import HTML, display
                jupyter_html = f'\n<div style="height: 400px; width: 100%; overflow-x: auto; overflow-y: hidden; resize: both">\
                                 \n<iframe src="{viewer_url}" style="width: 100%; height: 100%; border: none">\
                                 </iframe>\n</div>\n'
                display(HTML(jupyter_html))
            else:
                if Viewer.port_forwarding is None:
                    webbrowser.open(viewer_url, new=2, autoraise=True)
                else:
                    logging.warning(
                        "Impossible to open webbrowser through port forwarding. "\
                        "Either use Jupyter or open it manually.")

    @staticmethod
    def wait(require_client=False):
        """
        @brief Wait for all the meshes to finish loading in every clients.

        @param[in]  require_client   Wait for at least one client to be available
                                     before checking for mesh loading.
        """
        if Viewer.backend == 'gepetto-gui':
            return True  # Gepetto-gui is synchronous, so it cannot not be already loaded
        else:
            if Viewer._backend_proc is not None:
                if require_client:
                    Viewer._backend_obj.gui.wait()
                zmq_socket = Viewer._backend_obj.gui.window.zmq_socket
                def _is_loaded():
                    zmq_socket.send(b'meshes_loaded')
                    resp = zmq_socket.recv().decode("utf-8")
                    if resp:
                        return resp or min(np.array(resp.split(','), int)) == \
                            Viewer._backend_obj.info['nmeshes']
                    else:
                        return True
                while not _is_loaded():
                    time.sleep(0.1)
                return True
            else:
                raise NotImplementedError(
                    "Impossible to wait for mesh loading if the Meshcat server "\
                    "has not been opened by Python main thread for now.")

    def close(self=None):
        """
        @brief Close a given viewer instance, or all of them if no instance is specified.

        @remark Calling this method with an viewer instance always closes the client.
                It may also remove the robot from the server if the viewer attribute
                'delete_robot_on_close' is True.
                Moreover, it is the one having started the backend server, it also
                terminates it, resulting in closing every viewer somehow. It results
                in the same outcome than calling this method without specifying any
                viewer instance.
        """
        try:
            if self is None:
                self = Viewer
            else:
                Viewer._backend_robot_names.discard(self.robot_name)  # Consider that the robot name is now available, no matter whether the robot has actually been deleted or not
                if self.delete_robot_on_close:
                    self.delete_robot_on_close = False  # In case 'close' is called twice.
                    if Viewer.backend == 'gepetto-gui':
                        Viewer._delete_nodes_viewer(
                            [self.scene_name + '/' + self.robot_name])
                    else:
                        node_names = [
                            self._client.getViewerNodeName(
                                visual_obj, pin.GeometryType.VISUAL)
                            for visual_obj in self._rb.visual_model.geometryObjects]
                        Viewer._delete_nodes_viewer(node_names)
                        Viewer._backend_obj.info['nmeshes'] -= \
                            len(self._rb.visual_model.geometryObjects)
            if self == Viewer or self.is_backend_parent:
                self.is_backend_parent = False  # In case 'close' is called twice. No longer parent after closing.
                Viewer._backend_robot_names.clear()
                if self._backend_proc is not None and \
                        self._backend_proc.is_alive():
                    self._backend_proc.terminate()
                    self._backend_proc.join(timeout=0.5)
                    try:
                        backend_pid = Viewer._backend_proc.pid
                        proc = psutil.Process(backend_pid)
                        proc.send_signal(signal.SIGKILL)
                        os.waitpid(backend_pid, 0)  # Reap the zombies !
                    except psutil.NoSuchProcess:
                        pass
                    multiprocessing.active_children()
                if Viewer.backend == 'meshcat' and \
                        Viewer._backend_obj is not None and \
                        Viewer._backend_obj.browser is not None:
                    browser_proc = Viewer._backend_obj.browser._browser.process
                    Viewer._backend_obj.webui.close()
                    Viewer._backend_obj.browser.close()
                    browser_proc.kill()
                    try:
                        os.waitpid(browser_proc.pid, 0)
                        os.waitpid(os.getpid(), 0)
                    except ChildProcessError:
                        pass
                if self._backend_proc is Viewer._backend_proc:
                    Viewer._backend_obj = None
            if self._tempdir.startswith(tempfile.gettempdir()):
                try:
                    shutil.rmtree(self._tempdir)
                except FileNotFoundError:
                    pass
            self._backend_proc = None
        except AttributeError:
            pass

    @staticmethod
    def _is_notebook():
        """
        @brief Determine whether Python is running inside a Notebook or not.
        """
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type, if any
        except NameError:
            return False      # Probably standard Python interpreter

    @staticmethod
    def _get_colorized_urdf(urdf_path,
                            rgb,
                            mesh_root_path=None,
                            output_root_path=None):
        """
        @brief      Generate a unique colorized URDF.

        @remark     Multiple identical URDF model of different colors can be
                    loaded in Gepetto-viewer this way.

        @param[in]  urdf_path           Full path of the URDF file.
        @param[in]  rgb                 RGB code defining the color of the model. It is the same for each link.
        @param[in]  mesh_root_path      Root path of the meshes (optional).
        @param[in]  output_root_path    Root directory of the colorized URDF data (optional).

        @return     Full path of the colorized URDF file.
        """
        # Convert RGB array to string and xml tag
        color_string = "%.3f_%.3f_%.3f_1.0" % tuple(rgb)
        color_tag = "<color rgba=\"%.3f %.3f %.3f 1.0\"" % tuple(rgb)  # don't close tag with '>', in order to handle <color/> and <color></color>

        # Create the output directory
        if output_root_path is None:
            output_root_path = tempfile.mkdtemp()
        colorized_data_dir = os.path.join(
            output_root_path, "colorized_urdf_rgba_" + color_string)
        os.makedirs(colorized_data_dir, exist_ok=True)
        colorized_urdf_path = os.path.join(
            colorized_data_dir, os.path.basename(urdf_path))

        # Copy the meshes in the temporary directory, and update paths in URDF file
        with open(urdf_path, 'r') as urdf_file:
            colorized_contents = urdf_file.read()

        for mesh_fullpath in re.findall(
                '<mesh filename="(.*)"', colorized_contents):
            if mesh_root_path is not None:
                # Replace package path by mesh_root_path for convenience
                if mesh_fullpath.startswith('package://'):
                    mesh_fullpath = mesh_root_path + mesh_fullpath[9:]
            colorized_mesh_fullpath = os.path.join(
                colorized_data_dir, mesh_fullpath[1:])
            colorized_mesh_path = os.path.dirname(colorized_mesh_fullpath)
            if not os.access(colorized_mesh_path, os.F_OK):
                os.makedirs(colorized_mesh_path)
            shutil.copy2(mesh_fullpath, colorized_mesh_fullpath)
            colorized_contents = colorized_contents.replace(
                '"' + mesh_fullpath + '"', '"' + colorized_mesh_fullpath + '"', 1)
        colorized_contents = re.sub(
            r'<color rgba="[\d. ]*"', color_tag, colorized_contents)

        with open(colorized_urdf_path, 'w') as f:
            f.write(colorized_contents)

        return colorized_urdf_path

    @staticmethod
    def _urdf_fix_mesh_path(urdf_path, mesh_root_path, output_root_path=None):
        """
        @brief      Generate an URDF with updated mesh paths.

        @param[in]  urdf_path           Full path of the URDF file.
        @param[in]  mesh_root_path      Root path of the meshes (optional).
        @param[in]  output_root_path    Root directory of the fixed URDF file (optional).

        @return     Full path of the fixed URDF file.
        """
        # Extract all the mesh path that are not package path, continue if any
        with open(urdf_path, 'r') as urdf_file:
            urdf_contents = urdf_file.read()
        pathlists = [
            filename
            for filename in re.findall('<mesh filename="(.*)"', urdf_contents)
            if not filename.startswith('package://')]
        if not pathlists:
            return

        # If mesh root path already matching, then nothing to do
        mesh_root_path_orig = os.path.commonpath(pathlists)
        if mesh_root_path == mesh_root_path_orig:
            return urdf_path

        # Create the output directory
        if output_root_path is None:
            output_root_path = tempfile.mkdtemp()
        fixed_urdf_dir = os.path.join(output_root_path,
            "fixed_urdf" + mesh_root_path.replace('/', '_'))
        os.makedirs(fixed_urdf_dir, exist_ok=True)
        fixed_urdf_path = os.path.join(
            fixed_urdf_dir, os.path.basename(urdf_path))

        # Override the root mesh path with the desired one
        urdf_contents = urdf_contents.replace(
            mesh_root_path_orig, mesh_root_path)
        with open(fixed_urdf_path, 'w') as f:
            f.write(urdf_contents)

        return fixed_urdf_path

    @staticmethod
    def _get_client(start_if_needed=False,
                    close_at_exit=True,
                    timeout=2000):
        """
        @brief      Get a pointer to the running process of Gepetto-Viewer.

        @details    This method can be used to open a new process if necessary.
        .
        @param[in]  start_if_needed    Whether a new process must be created if
                                       no running process is found.
                                       Optional: False by default
        @param[in]  timeout            Wait some millisecond before considering
                                       starting new server has failed.
                                       Optional: 1s by default
        @param[in]  close_at_exit      Terminate backend server at Python exit.
                                       Optional: True by default

        @return     A pointer to the running Gepetto-viewer Client and its PID.
        """
        if Viewer.backend == 'gepetto-gui':
            from gepetto.corbaserver.client import Client as gepetto_client

            try:
                # Get the existing Gepetto client
                client = gepetto_client()
                # Try to fetch the list of scenes to make sure that the Gepetto client is responding
                client.gui.getSceneList()
                return client, None
            except Viewer._backend_exceptions:
                try:
                    client = gepetto_client()
                    client.gui.getSceneList()
                    return client, None
                except Viewer._backend_exceptions:
                    if start_if_needed:
                        FNULL = open(os.devnull, 'w')
                        client_proc = subprocess.Popen(
                            ['/opt/openrobots/bin/gepetto-gui'],
                            shell=False,
                            stdout=FNULL,
                            stderr=FNULL)
                        if close_at_exit:
                            atexit.register(Viewer.close)  # Cleanup at exit
                        for _ in range(max(2, int(timeout / 200))): # Must try at least twice for robustness
                            time.sleep(0.2)
                            try:
                                return gepetto_client(), client_proc
                            except Viewer._backend_exceptions:
                                pass
                        print("Impossible to open Gepetto-viewer")
            return None, None
        else:
            # Get the list of ports that are likely to correspond to meshcat servers
            meshcat_candidate_ports = []
            for conn in psutil.net_connections("tcp4"):
                if conn.status == 'LISTEN':
                    cmdline = psutil.Process(conn.pid).cmdline()
                    if 'python' in cmdline[0] or 'meshcat' in cmdline[-1]:
                        meshcat_candidate_ports.append(conn.laddr.port)

            # Use the first port responding to zmq request, if any
            zmq_url = None
            context = zmq.Context.instance()
            for port in meshcat_candidate_ports:
                try:
                    zmq_url = f"tcp://127.0.0.1:{port}"
                    zmq_socket = context.socket(zmq.REQ)
                    zmq_socket.RCVTIMEO = 50
                    zmq_socket.connect(zmq_url)
                    zmq_socket.send(b"url")
                    response = zmq_socket.recv().decode("utf-8")
                    if response[:4] != "http":
                        zmq_url = None
                except (zmq.error.Again, zmq.error.ZMQError):
                    zmq_url = None
                zmq_socket.close(linger=5)
                if zmq_url is not None:
                    break
            context.destroy(linger=5)

            # Launch a meshcat custom server if none has been found
            if zmq_url is None:
                proc, zmq_url, _ = start_zmq_server()
                if close_at_exit:
                    atexit.register(Viewer.close)  # Ensure proper cleanup at exit
            else:
                proc = None

            # Connect to the existing zmq server or create one if none.
            # Make sure the timeout is properly configured in any case
            # to avoid infinite waiting if case of closed server.
            with redirect_stdout(None):
                gui = meshcat.Visualizer(zmq_url)
            browser, webui = None, None

            class MeshcatWrapper:
                def __init__(self, gui, browser, webui):
                    self.gui = gui
                    self.browser = browser
                    self.webui = webui
                    self.info = {'nmeshes': 0}
            client = MeshcatWrapper(gui, browser, webui)

            return client, proc

    @staticmethod
    def _delete_nodes_viewer(nodes_path):
        """
        @brief      Delete a 'node' in Gepetto-viewer.

        @remark     Be careful, one must specify the full path of a node, including
                    all parent group, but without the window name, ie
                    'scene_name/robot_name' to delete the robot.

        @param[in]  nodes_path     Full path of the node to delete
        """
        try:
            if Viewer.backend == 'gepetto-gui':
                for node_path in nodes_path:
                    if node_path in Viewer._backend_obj.gui.getNodeList():
                        Viewer._backend_obj.gui.deleteNode(node_path, True)
            else:
                for node_path in nodes_path:
                    Viewer._backend_obj.gui[node_path].delete()
        except Viewer._backend_exceptions:
            pass

    def __getViewerNodeName(self, geometry_object, geometry_type):
        """
        @brief      Get the full path of a node associated with a given geometry
                    object and geometry type.

        @remark     This is a hidden function not intended to be called manually.

        @param[in]  geometry_object     Geometry object from which to get the node
        @param[in]  geometry_type       Geometry type. It must be either
                                        pin.GeometryType.VISUAL or pin.GeometryType.COLLISION
                                        for display and collision, respectively.

        @return     Full path of the associated node.
        """
        if geometry_type is pin.GeometryType.VISUAL:
            return self._rb.viz.viewerVisualGroupName + '/' + geometry_object.name
        elif geometry_type is pin.GeometryType.COLLISION:
            return self._rb.viz.viewerCollisionGroupName + '/' + geometry_object.name

    def __updateGeometryPlacements(self, visual=False):
        """
        @brief      Update the generalized position of a geometry object.

        @remark     This is a hidden function not intended to be called manually.

        @param[in]  visual      Wether it is a visual or collision update
        """
        if visual:
            geom_model = self._rb.visual_model
            geom_data = self._rb.visual_data
        else:
            geom_model = self._rb.collision_model
            geom_data = self._rb.collision_data
        pin.updateGeometryPlacements(self.pinocchio_model,
                                     self.pinocchio_data,
                                     geom_model, geom_data)

    def set_camera_transform(self, translation=None, rotation=None, relative=None):
        """
        @brief      Apply transform to the camera pose.

        @param[in]  translation    Position [X, Y, Z] as a list or 1D array
        @param[in]  rotation       Rotation [Roll, Pitch, Yaw] as a list or 1D np.array
        @param[in]  relative       How to apply the transform:
                                       - None: absolute
                                       - 'camera': relative to the current camera pose
                                       - other string: relative to a robot frame,
                                         not accounting for the rotation (traveling)
        """
        # Handling of translation and rotation arguments
        if not relative is None and relative != 'camera':
            if translation is None:
                translation = np.array([3.0, -3.0, 2.0])
            if rotation is None:
                rotation = np.array([1.3, 0.0, 0.8])
        else:
            if translation is None:
                translation = np.zeros(3)
            if rotation is None:
                rotation = np.zeros(3)
        rotation_mat = rpyToMatrix(np.asarray(rotation))
        translation = np.asarray(translation)

        # Compute the relative transformation if applicable
        if relative == 'camera':
            if Viewer.backend == 'gepetto-gui':
                H_orig = XYZQUATToSe3(
                    self._client.getCameraTransform(self._window_id))
            else:
                raise RuntimeError(
                    "'relative' = 'camera' option is not available in Meshcat.")
        elif relative is not None:
            # Get the body position, not taking into account the rotation
            body_id = self._rb.model.getFrameId(relative)
            try:
                body_transform = self._rb.data.oMf[body_id]
            except IndexError:
                raise ValueError("'relative' set to non existing frame.")
            H_orig = SE3(np.eye(3), body_transform.translation)

        # Perform the desired rotation
        if Viewer.backend == 'gepetto-gui':
            H_abs = SE3(rotation_mat, translation)
            if relative is None:
                self._client.setCameraTransform(
                    self._window_id, se3ToXYZQUAT(H_abs).tolist())
            else:
                # Not using recursive call for efficiency
                H_abs = H_abs * H_orig
                self._client.setCameraTransform(
                    self._window_id, se3ToXYZQUAT(H_abs).tolist())
        elif Viewer.backend == 'meshcat':
            if relative is None:
                # Meshcat camera is rotated by -pi/2 along Roll axis wrt the usual convention in robotics
                translation = CAMERA_INV_TRANSFORM_MESHCAT @ translation
                rotation = matrixToRpy(CAMERA_INV_TRANSFORM_MESHCAT @ rotation_mat)
                self._client.viewer["/Cameras/default/rotated/<object>"].\
                    set_transform(mtf.compose_matrix(
                        translate=translation, angles=rotation))
            else:
                H_abs = SE3(rotation_mat, translation)
                H_abs = H_abs * H_orig
                self.set_camera_transform(H_abs.translation, rotation)  # The original rotation is not modified

    def capture_frame(self, width=None, height=None, raw_data=False):
        """
        @brief      Take a snapshot and return associated data.

        @remark     This method is currently not available on Jupyter using
                    Meshcat backend because of asyncio conflict.

        @param[in]  width       Width for the image in pixels (not available with Gepetto-gui for now)
        @param[in]  height      Height for the image in pixels (not available with Gepetto-gui for now)
        @param[in]  raw_data    Whether to return a 2D numpy array, or the raw output
                                from the backend (the actual type may vary)
        """
        if Viewer.backend == 'gepetto-gui':
            if raw_data:
                raise ValueError(
                    "Raw data mode is not available using gepetto-gui.")
            if width is not None or height is None:
                logging.warning("Cannot specify window size using gepetto-gui.")
            with tempfile.NamedTemporaryFile(suffix=".png") as f:  # Gepetto is not able to save the frame if the file does not have ".png" extension
                self.save_frame(f.name)  # It is not possible to capture frame directly using gepetto-gui
                img_obj = Image.open(f.name)
                rgb_array = np.array(img_obj)[:, :, :-1]
            return rgb_array
        else:
            # Start rendering the viewer on host, in a hidden
            # Chromium browser, if not already started.
            if not Viewer._is_notebook():
                if Viewer._backend_obj.webui is None:
                    from requests_html import HTMLSession
                    Viewer._backend_obj.browser = HTMLSession()
                    Viewer._backend_obj.webui = Viewer._backend_obj.browser.get(
                        Viewer._backend_obj.gui.url())
                    Viewer._backend_obj.webui.html.render(
                        keep_page=True)
                    Viewer.wait(require_client=True)
            else:
                raise NotImplementedError(
                    "Capturing frame is not available in Jupyter for now.")

            # Send a javascript command to the hidden browser to
            # capture frame, then wait for it (since it is async).
            async def _capture_frame(client):
                if width is not None and height is not None:
                    await client.html.page.setViewport(
                        {'width': width, 'height': height})
                return await client.html.page.evaluate("""
                    () => {
                    return viewer.capture_image();
                    }
                """)
            loop = asyncio.get_event_loop()
            img_data_html = loop.run_until_complete(
                _capture_frame(Viewer._backend_obj.webui))

            # Parse the output to remove the html header, and
            # convert it into the desired output format.
            img_data = base64.decodebytes(str.encode(img_data_html[22:]))
            if raw_data:
                return img_data
            else:
                img_obj = Image.open(io.BytesIO(img_data))
                rgb_array = np.array(img_obj)[:, :, :-1]
                return rgb_array

    def save_frame(self, output_path, width=None, height=None):
        """
        @brief      Save a snapshot in png format.

        @remark     This method is currently not available on Jupyter using
                    Meshcat backend because of asyncio conflict.

        @param[in]  output_path    Fullpath of the image (.png extension is mandatory)
        @param[in]  width     Width for the image in pixels (not available with Gepetto-gui for now)
        @param[in]  height    Height for the image in pixels (not available with Gepetto-gui for now)
        """
        if not output_path.endswith('.png'):
            raise ValueError("The output path must have .png extension.")
        if Viewer.backend == 'gepetto-gui':
            self._client.captureFrame(self._window_id, output_path)
        else:
            img_data = self.capture_frame(width, height, True)
            with open(output_path, "wb") as f:
                f.write(img_data)

    def refresh(self, wait=False):
        """
        @brief      Refresh the configuration of Robot in the viewer.

        @param[in]  wait    Whether or not to wait for rendering to finish.
        """
        if Viewer._backend_obj is None or (self.is_backend_parent and
                not Viewer._backend_proc.is_alive()):
            raise RuntimeError(
                "No backend available. Please start one before calling this method.")

        with self._lock:
            if Viewer.backend == 'gepetto-gui':
                if self._rb.displayCollisions:
                    self._client.applyConfigurations(
                        [self.__getViewerNodeName(collision, pin.GeometryType.COLLISION)
                            for collision in self._rb.collision_model.geometryObjects],
                        [pin.se3ToXYZQUATtuple(self._rb.collision_data.oMg[\
                            self._rb.collision_model.getGeometryId(collision.name)])
                            for collision in self._rb.collision_model.geometryObjects]
                    )
                if self._rb.displayVisuals:
                    self.__updateGeometryPlacements(visual=True)
                    self._client.applyConfigurations(
                        [self.__getViewerNodeName(visual, pin.GeometryType.VISUAL)
                            for visual in self._rb.visual_model.geometryObjects],
                        [pin.se3ToXYZQUATtuple(self._rb.visual_data.oMg[\
                            self._rb.visual_model.getGeometryId(visual.name)])
                            for visual in self._rb.visual_model.geometryObjects]
                    )
                self._client.refresh()
            else:
                self.__updateGeometryPlacements(visual=True)
                for visual in self._rb.visual_model.geometryObjects:
                    T = self._rb.visual_data.oMg[\
                        self._rb.visual_model.getGeometryId(visual.name)].homogeneous
                    self._client.viewer[\
                        self.__getViewerNodeName(
                            visual, pin.GeometryType.VISUAL)].set_transform(T)
        if wait and Viewer.backend == 'meshcat':  # Gepetto-gui is already synchronous
            Viewer._backend_obj.gui.wait()

    def display(self, q, xyz_offset=None, wait=False):
        """
        @brief      Update the configuration of the robot.

        @details    Note that it will alter original robot data if viewer attribute
                    'use_theoretical_model' is false.

        @param[in]  q    Configuration of the robot, as a 1D numpy array.
        @param[in]  xyz_offset    Freeflyer position offset. (Note that it does not
                                  check for the robot actually have a freeflyer).
        @param[in]  wait    Whether or not to wait for rendering to finish.
        """
        if xyz_offset is not None:
            q = q.copy()  # Make a copy to avoid altering the original data
            q[:3] += xyz_offset

        with self._lock:
            if self._rb.model.nq != q.shape[0]:
                raise ValueError("The configuration vector does not have the right size.")
            self._rb.display(q)
        pin.framesForwardKinematics(self._rb.model, self._rb.data, q)  # This method is not called automatically by 'display' method
        if wait and Viewer.backend == 'meshcat':  # Gepetto-gui is already synchronous
            Viewer._backend_obj.gui.wait()

    def replay(self, evolution_robot, replay_speed, xyz_offset=None, wait=False):
        """
        @brief      Replay a complete robot trajectory at a given real-time ratio.

        @details    Note that it will alter original robot data if viewer attribute
                    'use_theoretical_model' is false.

        @param[in]  evolution_robot    list of State object of increasing time
        @param[in]  replay_speed       Real-time ratio
        @param[in]  xyz_offset         Freeflyer position offset. (Note that it does not
                                       check for the robot actually have a freeflyer).
        @param[in]  wait               Whether or not to wait for rendering to finish.
        """
        t = [s.t for s in evolution_robot]
        i = 0
        init_time = time.time()
        while i < len(evolution_robot):
            s = evolution_robot[i]
            try:
                self.display(s.q, xyz_offset, wait)
                wait = False  # It is enough to wait for the first timestep
            except Viewer._backend_exceptions:
                break
            t_simu = (time.time() - init_time) * replay_speed
            i = bisect_right(t, t_simu)
            sleep(s.t - t_simu)


def extract_viewer_data_from_log(log_data, robot):
    """
    @brief      Extract the minimal required information from raw log data in
                order to replay the simulation in a viewer.

    @details    It extracts the time and joint positions evolution.
    .
    @remark     Note that the quaternion angular velocity vectors are expressed
                it body frame rather than world frame.

    @param[in]  log_data    Data from the log file, in a dictionnary.
    @param[in]  robot       Jiminy robot.

    @return     Trajectory dictionary. The actual trajectory corresponds to
                the field "evolution_robot" and it is a list of State object.
                The other fields are additional information.
    """

    # Get the current robot model options
    model_options = robot.get_model_options()

    # Extract the joint positions time evolution
    t = log_data["Global.Time"]
    try:
        qe = np.stack([log_data["HighLevelController." + s]
                       for s in robot.logfile_position_headers], axis=-1)
    except KeyError:
        model_options['dynamics']['enableFlexibleModel'] = not robot.is_flexible
        robot.set_model_options(model_options)
        qe = np.stack([log_data["HighLevelController." + s]
                       for s in robot.logfile_position_headers], axis=-1)

    # Determine whether the theoretical model of the flexible one must be used
    use_theoretical_model = not robot.is_flexible

    # Make sure that the flexibilities are enabled
    model_options['dynamics']['enableFlexibleModel'] = True
    robot.set_model_options(model_options)

    # Create state sequence
    evolution_robot = []
    for i in range(len(t)):
        evolution_robot.append(State(t=t[i], q=qe[i]))

    return {'evolution_robot': evolution_robot,
            'robot': robot,
            'use_theoretical_model': use_theoretical_model}

def play_trajectories(trajectory_data,
                      mesh_root_path=None,
                      replay_speed=1.0,
                      record_video=False,
                      output_directory=None,
                      reference_link=None,
                      viewers=None,
                      start_paused=False,
                      wait_for_client=True,
                      camera_xyzrpy=None,
                      xyz_offset=None,
                      urdf_rgba=None,
                      backend=None,
                      window_name='python-pinocchio',
                      scene_name='world',
                      close_backend=None,
                      delete_robot_on_close=True,
                      verbose=True):
    """!
    @brief      Replay one or several robot trajectories in a viewer.

    @details    The ratio between the replay and the simulation time is kept constant to the desired ratio.
                One can choose between several backend (gepetto-gui or meshcat).

    @remark     The speed is independent of the plateform and the CPU power.

    @param[in]  trajectory_data     List of trajectory dictionary with keys:
                                    'evolution_robot': list of State object of increasing time
                                    'robot': jiminy robot (None if omitted)
                                    'use_theoretical_model':  whether to use the theoretical or actual model
    @param[in]  mesh_root_path      Optional, path to the folder containing the URDF meshes.
    @param[in]  replay_speed        Speed ratio of the simulation
    @param[in]  record_video        Whether or not to generate a video. For now, if this mode
                                    is enabled, one must make sure that the time evolution of
                                    each trajectories are the same, using a constant timestep.
    @param[in]  output_directory    Output directory where to save generated data,
                                    for instance the video if 'record_video' option is enabled.
    @param[in]  viewers             Optional, already instantiated viewers, associated one by one
                                    in order to each trajectory data.
    @param[in]  start_paused        Start the simulation is pause, waiting for keyboard input before
                                    starting to play the trajectories.
    @param[in]  wait_for_client     Wait for the client to finish loading the meshes before starting
    @param[in]  camera_xyzrpy       Absolute pose of the camera during replay (disable during video recording).
    @param[in]  xyz_offset          Constant translation of the root joint in world frame (1D numpy array).
    @param[in]  urdf_rgba           RGBA code defining the color of the model. It is the same for each link.
                                    Optional: Original colors of each link. No alpha.
    @param[in]  backend             Backend, one of 'meshcat' or 'gepetto-gui'. By default 'meshcat' is used
                                    in notebook environment and 'gepetto-gui' otherwise.
    @param[in]  window_name         Name of the Gepetto-viewer's window in which to display the robot.
                                    Optional: Common default name if omitted.
    @param[in]  scene_name          Name of the Gepetto-viewer's scene in which to display the robot.
                                    Optional: Common default name if omitted.
    @param[in]  close_backend       Close backend automatically at exit.
                                    Optional: Enable by default if not (presumably) available beforehand.
    @param[in]  delete_robot_on_close    Whether or not to delete the robot from the viewer when closing it.
    @param[in]  verbose              Add information to keep track of the process.

    @return     The viewers used to play the trajectories.
    """
    if viewers is not None:
        # Make sure that viewers is a list
        if not isinstance(viewers, list):
            viewers = [viewers]

        # Make sure the viewers are still running if specified
        if Viewer._backend_obj is None:
            viewers = None
        for viewer in viewers:
            if viewer.is_backend_parent and \
                    not Viewer._backend_proc.is_alive():
                viewers = None
                break

        # Do not close backend by default if it was supposed to be available
        if close_backend is None:
            close_backend = False

    if viewers is None:
        # Create new viewer instances
        viewers = []
        lock = Lock()
        for i in range(len(trajectory_data)):
            # Create a new viewer instance, and load the robot in it
            robot = trajectory_data[i]['robot']
            robot_name = "_".join(("robot",  str(i)))
            use_theoretical_model = trajectory_data[i]['use_theoretical_model']
            viewer = Viewer(
                robot,
                use_theoretical_model=use_theoretical_model,
                mesh_root_path=mesh_root_path,
                urdf_rgba=urdf_rgba[i] if urdf_rgba is not None else None,
                robot_name=robot_name,
                lock=lock,
                backend=backend,
                window_name=window_name,
                scene_name=scene_name,
                delete_robot_on_close=delete_robot_on_close)
            viewers.append(viewer)

            # Close backend by default
            if close_backend is None:
                close_backend = True

    # Set camera pose if requested
    if camera_xyzrpy is not None:
        viewers[0].set_camera_transform(
            translation=camera_xyzrpy[:3], rotation=camera_xyzrpy[3:])

    # Load robots in gepetto viewer
    if xyz_offset is None:
        xyz_offset = len(trajectory_data) * (None,)

    for i in range(len(trajectory_data)):
        try:
            viewers[i].display(
                trajectory_data[i]['evolution_robot'][0].q, xyz_offset[i])
        except Viewer._backend_exceptions:
            break
    if verbose:
        if backend == 'meshcat':
            print("Waiting for meshcat client in browser to connect: "\
                  f"{Viewer._get_client_url()}")
    Viewer.wait(require_client=(not record_video))  # Wait for the meshes to finish loading

    # Handle start-in-pause mode
    if start_paused and not Viewer._is_notebook():
        input("Press Enter to continue...")

    # Replay the trajectory
    if record_video:
        # Play trajectories without multithreading and record_video
        import cv2
        if verbose:
            print("Beginning video recording...")
        try:
            os.makedirs(output_directory)
        except OSError:
            pass
        img_array = []
        for i in tqdm(range(len(trajectory_data[0]['evolution_robot'])),
                      desc="Loading frames"):
            for j in range(len(trajectory_data)):
                viewers[j].display(trajectory_data[j]['evolution_robot'][i].q)
            viewers[0].set_camera_transform(relative=reference_link)
            frame = viewers[0].capture_frame(width=1000, height=1000)
            img_array.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        subsampling_rate = trajectory_data[0]['evolution_robot'][1].t \
                         - trajectory_data[0]['evolution_robot'][0].t
        if verbose:
            print(f"The subsampling rate is: {subsampling_rate}")
        video_fullpath = os.path.join(output_directory, "record_video.avi")
        out = cv2.VideoWriter(video_fullpath,
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              fps=1000/subsampling_rate,
                              frameSize=np.shape(img_array[0])[::-1])
        for i in tqdm(range(len(img_array)), desc="Writing frames"):
            out.write(img_array[i])
        out.release()
        if verbose:
            print(f"Video output to: {video_fullpath}")
    else:
        # Play trajectories with multithreading
        threads = []
        for i in range(len(trajectory_data)):
            threads.append(Thread(target=viewers[i].replay,
                                  args=(trajectory_data[i]['evolution_robot'],
                                        replay_speed, xyz_offset[i])))
        for i in range(len(trajectory_data)):
            threads[i].daemon = True
            threads[i].start()

        try:
            for i in range(len(trajectory_data)):
                threads[i].join()
        except KeyboardInterrupt:
            pass

    # Close backend if needed
    if close_backend:
        for viewer in viewers:
            viewer.close()

    return viewers

def play_logfiles(robots, logs_data, **kwargs):
    """
    @brief Play the content of a logfile in a viewer.
    @details This method simply formats the data then calls play_trajectories.

    @param robots    jiminy.Robot: either a single robot, or a list of robot for each log data.
    @param logs_data Either a single dictionary, or a list of dictionaries of simulation data log.
    @param kwargs    Keyword arguments for play_trajectories method.
    """
    # Reformat everything as lists
    if not isinstance(logs_data, list):
        logs_data = [logs_data]
    if not isinstance(robots, list):
        robots = [robots] * len(logs_data)

    # For each pair (log, robot), extract a trajectory object for play_trajectories
    trajectories = [extract_viewer_data_from_log(log, robot)
                    for log, robot in zip(logs_data, robots)]

    # Finally, play the trajectories
    return play_trajectories(trajectories, **kwargs)
