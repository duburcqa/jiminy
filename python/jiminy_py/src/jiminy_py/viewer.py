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
import asyncio
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
    VIEWER_ROOT, StaticFileHandlerNoCache, ZMQWebSocketBridge, WebSocketHandler)
from requests_html import HTMLSession

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio import SE3, se3ToXYZQUAT, XYZQUATToSe3
from pinocchio.rpy import rpyToMatrix, matrixToRpy

from .state import State


# Determine if the various backends are available
backends_available = ['meshcat']
import platform
if platform.system() == 'Linux':
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
        @brief   Function to provide cross-plateform time sleep with maximum accuracy.

        @param   dt   sleep duration in seconds.

        @details Use this method with cautious since it relies on busy looping principle instead of system scheduler.
                 As a result, it wastes a lot more resources than time.sleep. However, it is the only way to ensure
                 accurate delay on a non-real-time systems such as Windows 10.
    """
    _ = time.perf_counter() + dt
    while time.perf_counter() < _:
        pass


class Viewer:
    backend = None
    port_forwarding = None
    _backend_obj = None
    _backend_exceptions = ()
    _backend_proc = None
    _lock = Lock() # Unique threading.Lock for every simulations (in the same thread ONLY!)

    def __init__(self,
                 robot,
                 use_theoretical_model=False,
                 mesh_root_path=None,
                 urdf_rgba=None,
                 lock=None,
                 backend=None,
                 delete_robot_on_close=False,
                 robot_name="robot",
                 window_name='jiminy',
                 scene_name='world'):
        """
        @brief Constructor.

        @param robot          The jiminy.Robot to display.
        @param use_theoretical_model   Whether to use the theoretical (rigid) model or the flexible model for this robot.
        @param mesh_root_path Optional, path to the folder containing the URDF meshes.
        @param urdf_rgba      Color to use to display this robot (rgba).
-       @param lock           Custom threading.Lock
-                             Optional: Only required for parallel rendering using multiprocessing.
                              It is required since some backends does not support multiple
                              simultaneous connections (e.g. corbasever).
        @param backend        The name of the desired backend to use for rendering.
                              Optional, either 'gepetto-gui' or 'meshcat' ('panda3d' available soon).
        @param delete_robot_on_close     Enable automatic deletion of the robot when closing.
        @param robot_name     Unique robot name, to identify each robot in the viewer.
        @param scene_name     Scene name, used only when gepetto-gui is used as backend.
        @param window_name    Window name, used only when gepetto-gui is used as backend.
                              Note that it is not allowed to be equal to the window name.
        """
        # Backup some user arguments
        self.urdf_path = robot.urdf_path
        self.robot_name = robot_name
        self.scene_name = scene_name
        self.window_name = window_name
        self.use_theoretical_model = use_theoretical_model
        self._lock = lock if lock is not None else Viewer._lock
        self.delete_robot_on_close = delete_robot_on_close

        if self.scene_name == self.window_name:
            raise ValueError("Please, choose a different name for the scene and the window.")

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
                if Viewer._is_notebook() or (not 'gepetto-gui' in backends_available):
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
                        Viewer._get_client(True)
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
                    self._window_id = int(np.where(
                        [name == window_name for name in self._client.getWindowList()])[0][0])
            else:
                from pinocchio.visualize import MeshcatVisualizer
                from pinocchio.shortcuts import createDatas

                if Viewer._backend_obj is None:
                    Viewer._backend_obj, Viewer._backend_proc = \
                        Viewer._get_client(True)
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
            self._delete_nodes_viewer([scene_name + '/' + self.robot_name])
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

        # Refresh the viewer since the position of the meshes is not initialized at this point
        self.refresh()

    def __del__(self):
        self.close()

    @staticmethod
    def reset_port_forwarding(port_forwarding=None):
        Viewer.port_forwarding = port_forwarding

    @staticmethod
    def open_gui(create_if_needed=False):
        if Viewer.backend == 'meshcat':
            if Viewer._backend_obj is None:
                if create_if_needed:
                    Viewer._backend_obj, Viewer._backend_proc = \
                        Viewer._get_client(True)
                else:
                    raise RuntimeError(
                        "No meshcat backend available and 'create_if_needed' is set to False.")
            if Viewer._is_notebook() and Viewer.port_forwarding is not None:
                raise("Impossible to open web browser programmatically for Meshcat "\
                      "through port forwarding. Either use Jupyter or open it manually.")

            viewer_url = Viewer._backend_obj.gui.url()
            if Viewer.port_forwarding is not None:
                url_port_pattern = '(?<=:)[0-9]+(?=/)'
                port_localhost = int(re.search(url_port_pattern, viewer_url).group())
                assert port_localhost in Viewer.port_forwarding.keys(), \
                    "Port forwarding defined but no port mapping associated with {port_localhost}."
                port_remote = Viewer.port_forwarding[port_localhost]
                viewer_url = re.sub(url_port_pattern, str(port_remote), viewer_url)

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
                    logging.warning("Impossible to open webbrowser through port forwarding. "\
                                    "Either use Jupyter or open it manually.")
        else:
            raise RuntimeError("Showing client is only available using 'meshcat' backend.")

    def delete_robot(self):
        try:
            if Viewer.backend == 'gepetto-gui':
                self._delete_nodes_viewer(
                    [self.scene_name + '/' + self.robot_name])
            else:
                node_names = [
                    self._client.getViewerNodeName(
                        visual_obj, pin.GeometryType.VISUAL)
                    for visual_obj in self._rb.visual_model.geometryObjects]
                self._delete_nodes_viewer(node_names)
        except AttributeError:
            pass

    def close(self=None):
        if self is None:
            self = Viewer
        else:
            if self.delete_robot_on_close:
                self.delete_robot()
        if self == Viewer or self.is_backend_parent:
            if Viewer._backend_proc is not None and Viewer._backend_proc.is_alive():
                Viewer._backend_proc.terminate()
                Viewer._backend_proc.join(timeout=0.5)
                try:
                    proc = psutil.Process(Viewer._backend_proc.pid)
                    proc.send_signal(signal.SIGKILL)
                    os.waitpid(Viewer._backend_proc.pid, 0)  # Reap the zombies !
                except psutil.NoSuchProcess:
                    pass
                multiprocessing.active_children()
            if Viewer.backend == 'meshcat' and Viewer._backend_obj is not None and \
                    Viewer._backend_obj.browser is not None:
                Viewer._backend_obj.webui.close()
                Viewer._backend_obj.browser.close()
                Viewer._backend_obj.browser._browser.process.kill()
                try:
                    os.waitpid(Viewer._backend_obj.browser._browser.process.pid, 0)
                    os.waitpid(os.getpid(), 0)
                except ChildProcessError:
                    pass
        if self._backend_proc is Viewer._backend_proc:
            Viewer._backend_obj = None
        if self._tempdir.startswith(tempfile.gettempdir()):
            shutil.rmtree(self._tempdir)
        self._backend_proc = None

    @staticmethod
    def _is_notebook():
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
    def _get_colorized_urdf(urdf_path, rgb, mesh_root_path=None, output_root_path=None):
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
    def _get_client(create_if_needed=False, create_timeout=2000):
        """
        @brief      Get a pointer to the running process of Gepetto-Viewer.

        @details    This method can be used to open a new process if necessary.
        .
        @param[in]  create_if_needed    Whether a new process must be created if
                                        no running process is found.
                                        Optional: False by default
        @param[in]  create_timeout      Wait some millisecond before considering
                                        creating new viewer as failed.
                                        Optional: 1s by default

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
            except:
                try:
                    client = gepetto_client()
                    client.gui.getSceneList()
                    return client, None
                except:
                    if create_if_needed:
                        FNULL = open(os.devnull, 'w')
                        client_proc = subprocess.Popen(
                            ['/opt/openrobots/bin/gepetto-gui'],
                            shell=False,
                            stdout=FNULL,
                            stderr=FNULL)
                        atexit.register(Viewer.close)  # Cleanup at exit
                        for _ in range(max(2, int(create_timeout / 200))): # Must try at least twice for robustness
                            time.sleep(0.2)
                            try:
                                return gepetto_client(), client_proc
                            except:
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
                class MyFileHandler(StaticFileHandlerNoCache):
                    def initialize(self, default_path, default_filename, fallback_path):
                        self.default_path = os.path.abspath(default_path)
                        self.default_filename = default_filename
                        self.fallback_path = os.path.abspath(fallback_path)
                        super().initialize(self.default_path, self.default_filename)
                    def validate_absolute_path(self, root, absolute_path):
                        if os.path.exists(absolute_path) and \
                                os.path.basename(absolute_path) != 'index.html' :
                            return super().validate_absolute_path(root, absolute_path)
                        else:
                            return os.path.join(
                                self.fallback_path, absolute_path[(len(root)+1):])
                def make_app(self):
                    return tornado.web.Application([
                        (r"/static/(.*)", MyFileHandler, {
                            "default_path": VIEWER_ROOT,
                            "fallback_path": os.path.join(os.path.dirname(__file__), "meshcat"),
                            "default_filename": "index.html"}),
                        (r"/", WebSocketHandler, {"bridge": self})
                    ])
                ZMQWebSocketBridge.make_app = make_app

                # Meshcat server deamon, using in/out argument to get
                # the zmq url instead of reading stdout as it was.
                def meshcat_zmqserver(zmq_url):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    with open(os.devnull, 'w') as f:
                        with redirect_stderr(f):
                            bridge = ZMQWebSocketBridge()
                            info['zmq_url'] = bridge.zmq_url
                            try:
                                bridge.run()
                            except KeyboardInterrupt:
                                pass

                # Run meshcat server in background using multiprocessing
                # Process to enable monkey patching and proper interprocess
                # communication through a manager.
                manager = multiprocessing.Manager()
                info = manager.dict()
                proc = multiprocessing.Process(target=meshcat_zmqserver, args=(info,))
                proc.daemon = False
                proc.start()
                atexit.register(Viewer.close)  # Cleanup at exit

                # Wait for the process to finish initialization
                while not info:
                    pass
                zmq_url = info['zmq_url']
            else:
                proc = None

            # Connect to the existing zmq server or create one if none.
            # Make sure the timeout is properly configured in any case
            # to avoid infinite waiting if case of closed server.
            with redirect_stdout(None):
                gui = meshcat.Visualizer(zmq_url)
                gui.window.zmq_socket.RCVTIMEO = 50
                if not Viewer._is_notebook():
                    browser = HTMLSession()
                    webui = browser.get(gui.url() + "index.html")
                    webui.html.render(keep_page=True, sleep=0.5)  # Must be long enough to render all bodies
                else:
                    browser, webui = None, None

            class MeshcatWrapper:
                def __init__(self, gui, browser, webui):
                    self.gui = gui
                    self.browser = browser
                    self.webui = webui
            client = MeshcatWrapper(gui, browser, webui)

            return client, proc

    def _delete_nodes_viewer(self, nodes_path):
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
        # translation : [Px, Py, Pz], rotation : [Roll, Pitch, Yaw]
        # If no translation or rotation are set, initialize camera towards origin of the plane

        # Handling of default transformation
        if translation is None and rotation is None:
            translation = np.array([3.0, -3.0, 2.0])
            rotation = np.array([1.3, 0.0, 0.8])
        if translation is None:
            translation = np.zeros(3)
        if rotation is None:
            rotation = np.zeros(3)

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
            H_abs = SE3(rpyToMatrix(np.asarray(rotation)), np.asarray(translation))
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
                rotation = matrixToRpy(CAMERA_INV_TRANSFORM_MESHCAT @ rpyToMatrix(rotation))
                self._client.viewer["/Cameras/default/rotated/<object>"].set_transform(
                    mtf.compose_matrix(translate=translation, angles=rotation))
            else:
                H_abs = SE3(rpyToMatrix(np.asarray(rotation)), np.asarray(translation))
                H_abs = H_abs * H_orig
                self.set_camera_transform(H_abs.translation, rotation)  # The original rotation is not modified

    def capture_frame(self, width=None, height=None, raw_data=False):
        if Viewer.backend == 'gepetto-gui':
            assert not raw_data, "Raw data mode is not available using gepetto-gui."
            if width is not None or height is None:
                logging.warning("Cannot specify window size using gepetto-gui.")
            with tempfile.NamedTemporaryFile(suffix=".png") as f:  # Gepetto is not able to capture the frame if the file does not have ".png" extension
                self.save_frame(f.name)  # It is not possible to capture frame directly using gepetto-gui
                img_obj = Image.open(f.name)
                rgb_array = np.array(img_obj)[:, :, :-1]
            return rgb_array
        else:
            assert Viewer._backend_obj.webui is not None, \
                "Capturing frame is not available in Jupyter for now."
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
            img_data = base64.decodebytes(str.encode(img_data_html[22:]))
            if raw_data:
                return img_data
            else:
                img_obj = Image.open(io.BytesIO(img_data))
                rgb_array = np.array(img_obj)[:, :, :-1]
                return rgb_array

    def save_frame(self, output_path, width=None, height=None):
        if Viewer.backend == 'gepetto-gui':
            self._client.captureFrame(self._window_id, output_path)
        else:
            img_data = self.capture_frame(width, height, True)
            with open(output_path, "wb") as f:
                f.write(img_data)

    def refresh(self):
        """
        @brief      Refresh the configuration of Robot in the viewer.
        """
        if Viewer._backend_obj is None or (self.is_backend_parent and
                not Viewer._backend_proc.is_alive()):
            raise RuntimeError(
                "No backend available. Please start one before calling this method.")

        if not self._lock.acquire(timeout=0.2):
            raise RuntimeError("Impossible to acquire backend lock.")

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

        self._lock.release()

    def display(self, q, xyz_offset=None):
        if xyz_offset is not None:
            q = q.copy() # Make sure to use a copy to avoid altering the original data
            q[:3] += xyz_offset

        if not self._lock.acquire(timeout=0.2):
            raise RuntimeError("Impossible to acquire backend lock.")
        self._rb.display(q)
        self._lock.release()
        pin.framesForwardKinematics(self._rb.model, self._rb.data, q)

    def replay(self, evolution_robot, replay_speed, xyz_offset=None):
        t = [s.t for s in evolution_robot]
        i = 0
        init_time = time.time()
        while i < len(evolution_robot):
            s = evolution_robot[i]
            try:
                self.display(s.q, xyz_offset)
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
                      viewers=None,
                      start_paused=False,
                      camera_xyzrpy=None,
                      xyz_offset=None,
                      urdf_rgba=None,
                      backend=None,
                      window_name='python-pinocchio',
                      scene_name='world',
                      close_backend=None,
                      delete_robot_on_close=True):
    """!
    @brief      Display a robot trajectory in a viewer.

    @details    The ratio between the replay and the simulation time is kept constant to the desired ratio.
                One can choose between several backend (gepetto-gui or meshcat).

    @remark     The speed is independent of the plateform and the CPU power.

    @param[in]  trajectory_data     Trajectory dictionary with keys:
                                    'evolution_robot': list of State object of increasing time
                                    'robot': jiminy robot (None if omitted)
                                    'use_theoretical_model':  whether the theoretical or actual model must be used
    @param[in]  mesh_root_path      Optional, path to the folder containing the URDF meshes.
    @param[in]  xyz_offset          Constant translation of the root joint in world frame (1D numpy array)
    @param[in]  urdf_rgba           RGBA code defining the color of the model. It is the same for each link.
                                    Optional: Original colors of each link. No alpha.
    @param[in]  replay_speed        Speed ratio of the simulation
    @param[in]  backend             Backend, one of 'meshcat' or 'gepetto-gui'. By default 'meshcat' is used
                                    in notebook environment and 'gepetto-gui' otherwise.
    @param[in]  window_name         Name of the Gepetto-viewer's window in which to display the robot.
                                    Optional: Common default name if omitted.
    @param[in]  scene_name          Name of the Gepetto-viewer's scene in which to display the robot.
                                    Optional: Common default name if omitted.
    @param[in] delete_robot_on_close    Whether or not to delete the robot from the viewer when closing it.

    @return     The viewers used to play the trajectories.
    """
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

            # Wait a few moment, to give enough time to load meshes if necessary
            time.sleep(0.5)

        if viewers[0].is_backend_parent:
            # Initialize camera pose
            if camera_xyzrpy is not None:
                viewers[0].set_camera_transform(
                    translation=camera_xyzrpy[:3], rotation=camera_xyzrpy[3:])

            # Close backend by default if it was not available beforehand
            if close_backend is None:
                close_backend = True
    else:
        # Make sure that viewers is a list
        if not isinstance(viewers, list):
            viewers = [viewers]

        # Make sure the viewers are still running
        is_backend_running = True
        for viewer in viewers:
            if viewer.is_backend_parent and not Viewer._backend_proc.is_alive():
                is_backend_running = False
        if Viewer._backend_obj is None:
            is_backend_running = False
        if not is_backend_running:
            raise RuntimeError("Viewers backend not available.")

    # Load robots in gepetto viewer
    if xyz_offset is None:
        xyz_offset = len(trajectory_data) * (None,)

    for i in range(len(trajectory_data)):
        if xyz_offset is not None and xyz_offset[i] is not None:
            q = trajectory_data[i]['evolution_robot'][0].q.copy()
            q[:3] += xyz_offset[i]
        else:
            q = trajectory_data[i]['evolution_robot'][0].q
        try:
            viewers[i]._rb.display(q)
        except Viewer._backend_exceptions:
            break

    # Handle start-in-pause mode
    if start_paused and not Viewer._is_notebook():
        input("Press Enter to continue...")

    # Replay the trajectory
    threads = []
    for i in range(len(trajectory_data)):
        threads.append(Thread(target=viewers[i].display,
                              args=(trajectory_data[i]['evolution_robot'],
                                    replay_speed, xyz_offset[i])))
    for i in range(len(trajectory_data)):
        threads[i].start()
    for i in range(len(trajectory_data)):
        threads[i].join()

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
