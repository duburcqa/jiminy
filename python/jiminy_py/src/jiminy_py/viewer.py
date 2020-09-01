#!/usr/bin/env python

## @file jiminy_py/viewer.py

import os
import re
import io
import time
import types
import psutil
import shutil
import signal
import base64
import atexit
import pathlib
import asyncio
import tempfile
import subprocess
import logging
import webbrowser
import numpy as np
import tornado.web
import multiprocessing
from tqdm import tqdm
from PIL import Image
from bisect import bisect_right
from threading import Thread, Lock
from scipy.interpolate import interp1d

import zmq
import meshcat.transformations as mtf

import pinocchio as pin
from pinocchio import SE3, se3ToXYZQUAT, XYZQUATToSe3
from pinocchio.rpy import rpyToMatrix, matrixToRpy
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
from pinocchio.shortcuts import createDatas

from .state import State
from .meshcat.wrapper import MeshcatWrapper, is_notebook


# Determine if the various backends are available
backends_available = ['meshcat']
if __import__('platform').system() == 'Linux':
    try:
        import gepetto as _gepetto
        import omniORB as _omniORB
        backends_available.append('gepetto-gui')
    except ImportError:
        pass

def default_backend():
    if is_notebook() or not 'gepetto-gui' in backends_available:
        return 'meshcat'
    else:
        return 'gepetto-gui'


# Create logger
class DuplicateFilter:
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv

logger = logging.getLogger(__name__)
logger.addFilter(DuplicateFilter())


CAMERA_INV_TRANSFORM_MESHCAT = rpyToMatrix(np.array([-np.pi/2, 0.0, 0.0]))
DEFAULT_CAMERA_XYZRPY = ([7.5, 0.0, 1.4], [1.4, 0.0, np.pi/2])
DEFAULT_CAPTURE_SIZE = 500
VIDEO_FRAMERATE = 30
VIDEO_SIZE = (1000, 1000)


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


class ProcessWrapper:
    """
    @brief     Wrap multiprocessing Process, subprocess Popen, and psutil
               Process in the same object to have the same user interface.

    @details   It also makes sure that the process is properly terminated
               at Python exits, and without zombies left behind.
    """
    def __init__(self, proc, kill_at_exit=False):
        self._proc = proc
        # Make sure the process is killed at Python exit
        if kill_at_exit:
            atexit.register(self.kill)

    def is_parent(self):
        return isinstance(self._proc, (
            subprocess.Popen, multiprocessing.Process))

    def is_alive(self):
        if isinstance(self._proc, subprocess.Popen):
            return self._proc.poll() is None
        elif isinstance(self._proc, multiprocessing.Process):
            return self._proc.is_alive()
        elif isinstance(self._proc, psutil.Process):
            return self._proc.status() in [
                psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]

    def wait(self, timeout=None):
        if isinstance(self._proc, multiprocessing.Process):
            return self._proc.join(timeout)
        elif isinstance(self._proc, (
                subprocess.Popen, psutil.Process)):
            return self._proc.wait(timeout)

    def kill(self):
        if self.is_parent() and self.is_alive():
            # Try to terminate cleanly
            self._proc.terminate()
            self.wait(timeout=0.5)

            # Force kill if necessary and reap the zombies
            try:
                psutil.Process(self._proc.pid).kill()
                os.waitpid(self._proc.pid, 0)
                os.waitpid(os.getpid(), 0)
            except (psutil.NoSuchProcess, ChildProcessError):
                pass
            multiprocessing.active_children()


class Viewer:
    backend = default_backend()
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
                 open_gui_if_parent=True,
                 delete_robot_on_close=False,
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
        @param open_gui_if_parent        Open GUI if new viewer's backend server is started.
        @param delete_robot_on_close     Enable automatic deletion of the robot when closing.
        @param robot_name     Unique robot name, to identify each robot in the viewer.
                              Optional: Randomly generated identifier by default.
        @param window_name    Window name, used only when gepetto-gui is used as backend.
                              Note that it is not allowed to be equal to the window name.
        @param scene_name     Scene name, used only when gepetto-gui is used as backend.
        """
        # Handling of default arguments
        if robot_name is None:
            uniq_id = next(tempfile._get_candidate_names())
            robot_name = "_".join(("robot", uniq_id))

        # Backup some user arguments
        self.urdf_path = os.path.realpath(robot.urdf_path)
        self.robot_name = robot_name
        self.scene_name = scene_name
        self.window_name = window_name
        self.use_theoretical_model = use_theoretical_model
        self._lock = lock if lock is not None else Viewer._lock
        self.delete_robot_on_close = delete_robot_on_close


        # Define camera update function, that will be called systematically
        # after calling refresh or update. It will be used later for enabling
        # to attach the camera to a given frame and automatically track it
        # without explicitly calling 'set_camera_transform'.
        self.detach_camera()

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
                if is_notebook() or not 'gepetto-gui' in backends_available:
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
            logging.warning("Different backend already running. Closing it...")
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
            if not Viewer.is_open():
                is_backend_running = False
            if Viewer.backend == 'gepetto-gui':
                try:
                    Viewer._backend_obj.gui.refresh()
                except Viewer._backend_exceptions:
                    is_backend_running = False
            if not is_backend_running:
                Viewer._backend_obj = None
                Viewer._backend_proc = None
                Viewer._backend_exception = None
        else:
            is_backend_running = False

        # Access the current backend or create one if none is available
        self.__is_open = False
        self.is_backend_parent = False
        try:
            if Viewer.backend == 'gepetto-gui':
                if Viewer._backend_obj is None:
                    Viewer._backend_obj, Viewer._backend_proc = \
                        Viewer.__get_client(start_if_needed=True)
                    self.is_backend_parent = Viewer._backend_proc.is_parent()
                self.__is_open = True
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
                if Viewer._backend_obj is None:
                    Viewer._backend_obj, Viewer._backend_proc = \
                        Viewer.__get_client(start_if_needed=True)
                    self.is_backend_parent = Viewer._backend_proc.is_parent()
                self.__is_open = True
                if not is_notebook():
                    if self.is_backend_parent and open_gui_if_parent:
                        self.open_gui()
                else:
                    # There is no display cell already opened. So opening one
                    # since it is probably what the user is expecting, but
                    # there is no fixed rule. Then, wait for it to open,
                    # otherwise we will get into trouble...
                    if Viewer._backend_obj.comm_manager.n_comm == 0:
                        self.open_gui()

                self._client = MeshcatVisualizer(self.pinocchio_model, None, None)
                self._client.viewer = Viewer._backend_obj.gui
        except Exception as e:
            raise RuntimeError("Impossible to create or connect to backend.") from e

        # Set the default camera pose if the viewer is not running before
        if self.is_backend_parent:
            self.set_camera_transform()

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
        Viewer._backend_robot_names.add(robot_name)

        # Refresh the viewer since the position of the meshes is not initialized at this point
        self.refresh()

    def __del__(self):
        """
        @brief Destructor.

        @remark It automatically close the viewer before being garbage collected.
        """
        self.close()

    def __must_be_open(fct):
        def fct_safe(*args, **kwargs):
            self = None
            if args and isinstance(args[0], Viewer):
                self = args[0]
            self = kwargs.get('self', self)
            if not Viewer.is_open(self):
                raise RuntimeError("No backend available. "\
                    f"Please start one before calling '{fct.__name__}'.")
            return fct(*args, **kwargs)
        return fct_safe

    @staticmethod
    def open_gui(start_if_needed=False):
        """
        @brief Open a new viewer graphical interface.

        @remark  This method is not supported by Gepetto-gui since it does not have a classical
                 server-client mechanism. One and only one graphical interface (client) can be
                 opened, and its lifetime is tied to the one of the server itself.
        """
        if Viewer.backend == 'gepetto-gui':
            raise RuntimeError(
                "Showing client is only available using 'meshcat' backend.")
        else:
            if Viewer._backend_obj is None:
                Viewer._backend_obj, Viewer._backend_proc = \
                    Viewer.__get_client(start_if_needed)

            if is_notebook():
                import urllib
                from IPython.core.display import HTML, display

                # Scrap the viewer html content, including javascript dependencies
                viewer_url = Viewer._backend_obj.gui.url()
                html_content = urllib.request.urlopen(viewer_url).read().decode()
                pattern = '<script type="text/javascript" src="%s"></script>'
                scripts_js = re.findall(pattern % '(.*)', html_content)
                for file in scripts_js:
                    file_path = os.path.join(viewer_url, file)
                    js_content = urllib.request.urlopen(file_path).read().decode()
                    html_content = html_content.replace(pattern % file, f"""
                    <script type="text/javascript">
                    {js_content}
                    </script>""")

                if is_notebook() == 1:
                    # Open it in a HTML iframe on Jupyter, since it is not
                    # possible to load it directly.
                    html_content = html_content.replace("\"", "&quot;").\
                        replace("'", "&apos;")
                    display(HTML(f"""
                        <div class="resizable" style="height: 400px; width: 100%; overflow-x: auto; overflow-y: hidden; resize: both">
                        <iframe srcdoc="{html_content}" style="width: 100%; height: 100%; border: none;"></iframe>
                        </div>
                    """))
                else:
                    # Adjust the initial window size
                    html_content = html_content.replace('<div id="meshcat-pane">',
                        '<div id="meshcat-pane" class="resizable" style="height: 400px; '\
                            'width: 100%; overflow-x: auto; overflow-y: hidden; resize: both">')
                    display(HTML(html_content))
            else:
                try:
                    webbrowser.get()
                    webbrowser.open(viewer_url, new=2, autoraise=True)
                except Exception:  # Fail if not browser is available
                    logger.warning("No browser available for display. Please install one manually.")
                    return  # Skip waiting since it is not possible in this case

            # Wait for the display to finish loading
            Viewer.wait(require_client=True)

    @staticmethod
    @__must_be_open
    def wait(require_client=False):
        """
        @brief Wait for all the meshes to finish loading in every clients.

        @param[in]  require_client   Wait for at least one client to be available
                                     before checking for mesh loading.
        """
        if Viewer.backend == 'gepetto-gui':
            return None  # Gepetto-gui is synchronous, so it cannot not be already loaded
        else:
            Viewer._backend_obj.wait(require_client)

    def is_open(self=None):
        is_open_ = Viewer._backend_proc is not None and \
            Viewer._backend_proc.is_alive()
        if self is not None:
            is_open_ = is_open_ and self.__is_open
        return is_open_

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
            if Viewer.backend == 'meshcat' and Viewer._backend_obj is not None:
                Viewer._backend_obj.gui.window.zmq_socket.RCVTIMEO = 50
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
            if self == Viewer:  # NEVER closing backend if closing instances, even for the parent. It will be closed at Python exit automatically.
                Viewer._backend_robot_names.clear()
                if Viewer.backend == 'meshcat' and \
                        Viewer._backend_obj is not None:
                    Viewer._backend_obj.close()
                    ProcessWrapper(Viewer._backend_obj.recorder.proc).kill()
                if Viewer.is_open():
                    Viewer._backend_proc.kill()
                Viewer._backend_obj = None
                Viewer._backend_proc = None
            else:
                self.__is_open = False
            if self._tempdir.startswith(tempfile.gettempdir()):
                try:
                    shutil.rmtree(self._tempdir)
                except FileNotFoundError:
                    pass
            if Viewer.backend == 'meshcat' and Viewer._backend_obj is not None:
                Viewer._backend_obj.gui.window.zmq_socket.RCVTIMEO = -1
        except Exception:  # Catch everything, since we do not want this method to fail in any circumstances
            pass

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
            return urdf_path

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
    def __get_client(start_if_needed=False,
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

            def _gepetto_client_connect(get_proc_info=False):
                nonlocal close_at_exit

                # Get the existing Gepetto client
                client = gepetto_client()

                # Try to fetch the list of scenes to make sure that the Gepetto client is responding
                client.gui.getSceneList()

                # Get the associated process information if requested
                if not get_proc_info:
                    return client
                proc = [p for p in psutil.process_iter()
                        if 'gepetto-gui' in p.cmdline()[0]][0]
                return client, ProcessWrapper(proc, close_at_exit)

            try:
                return _gepetto_client_connect(get_proc_info=True)
            except Viewer._backend_exceptions:
                try:
                    return _gepetto_client_connect(get_proc_info=True)
                except Viewer._backend_exceptions:
                    if start_if_needed:
                        FNULL = open(os.devnull, 'w')
                        proc = subprocess.Popen(['gepetto-gui'],
                            shell=False, stdout=FNULL, stderr=FNULL)
                        proc = ProcessWrapper(proc, close_at_exit)
                        for _ in range(max(2, int(timeout / 200))): # Must try at least twice for robustness
                            time.sleep(0.2)
                            try:
                                return _gepetto_client_connect(), proc
                            except Viewer._backend_exceptions:
                                pass
                        raise RuntimeError("Impossible to open Gepetto-viewer.")
            return None, None
        else:
            # Get the list of connections that are likely to correspond to meshcat servers
            meshcat_candidate_conn = []
            for conn in psutil.net_connections("tcp4"):
                if conn.status == 'LISTEN' and conn.laddr.ip == '127.0.0.1':
                    cmdline = psutil.Process(conn.pid).cmdline()
                    if 'python' in cmdline[0] or 'meshcat' in cmdline[-1]:
                        meshcat_candidate_conn.append(conn)

            # Exclude ipython kernel ports from the look up because sending a
            # message on ipython ports will throw a low-level exception, that
            # is not blocking on Jupyter, but is on Google Colab.
            excluded_ports = []
            if is_notebook():
                try:
                    excluded_ports += list(
                        get_ipython().kernel._recorded_ports.values())
                except (NameError, AttributeError):
                    pass  # No Ipython kernel running

            # Use the first port responding to zmq request, if any
            zmq_url = None
            context = zmq.Context.instance()
            for conn in meshcat_candidate_conn:
                try:
                    # Note that the timeout must be long enough to give enough
                    # time to the server to respond, but not to long to avoid
                    # sending to much time spanning the available connections.
                    port = conn.laddr.port
                    if port in excluded_ports:
                        continue
                    zmq_url = f"tcp://127.0.0.1:{port}"
                    zmq_socket = context.socket(zmq.REQ)
                    zmq_socket.RCVTIMEO = 250  # millisecond
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

            # Create a meshcat server if needed and connect to it
            client = MeshcatWrapper(zmq_url)
            if client.server_proc is None:
                proc = psutil.Process(conn.pid)
            else:
                proc = client.server_proc
            proc = ProcessWrapper(proc, close_at_exit)

            return client, proc

    @staticmethod
    @__must_be_open
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

    @__must_be_open
    def set_camera_transform(self, translation=None, rotation=None, relative=None):
        """
        @brief      Apply transform to the camera pose.

        @param[in]  translation    Position [X, Y, Z] as a list or 1D array
        @param[in]  rotation       Rotation [Roll, Pitch, Yaw] as a list or 1D np.array
        @param[in]  relative       How to apply the transform:
                                       - None: absolute
                                       - 'camera': relative to the current camera pose
                                       - other string: relative to a robot frame,
                                         not accounting for the rotation (travelling)
        """
        # Handling of translation and rotation arguments
        if not relative is None and relative != 'camera':
            if translation is None:
                translation = np.array([3.0, -3.0, 1.0])
            if rotation is None:
                rotation = np.array([1.3, 0.0, 0.8])
        else:
            if translation is None:
                translation = DEFAULT_CAMERA_XYZRPY[0]
            if rotation is None:
                rotation = DEFAULT_CAMERA_XYZRPY[1]
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
                H_abs = H_orig * H_abs
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
                H_abs = H_orig * H_abs
                self.set_camera_transform(H_abs.translation, rotation)  # The original rotation is not modified

    def attach_camera(self, frame=None, translation=None, rotation=None):
        """
        @brief      Attach the camera to a given robot frame.

        @details    Only the position of the frame is taken into account.
                    A custom relative pose of the camera wrt to the frame
                    can be further specified.

        @param[in]  frame          Frame of the robot to follow with the camera.
        @param[in]  translation    Relative position of the camera wrt to the frame [X, Y, Z]
        @param[in]  rotation       Relative rotation of the camera wrt to the frame [Roll, Pitch, Yaw]
        """
        def __update_camera_transform(self):
            nonlocal frame, translation, rotation
            self.set_camera_transform(translation, rotation, relative=frame)
        self.__update_camera_transform = types.MethodType(
            __update_camera_transform, self)

    def detach_camera(self):
        """
        @brief      Detach the camera.

        @details    Must be called to undo 'attach_camera', so that it will
                    stop automatically tracking a frame.
        """
        self.__update_camera_transform = lambda : None

    @__must_be_open
    def capture_frame(self,
                      width=DEFAULT_CAPTURE_SIZE,
                      height=DEFAULT_CAPTURE_SIZE,
                      raw_data=False):
        """
        @brief      Take a snapshot and return associated data.

        @param[in]  width       Width for the image in pixels (not available with Gepetto-gui for now).
                                Optional: DEFAULT_CAPTURE_SIZE by default. None to keep unchanged.
        @param[in]  height      Height for the image in pixels (not available with Gepetto-gui for now).
                                Optional: DEFAULT_CAPTURE_SIZE by default. None to keep unchanged.
        @param[in]  raw_data    Whether to return a 2D numpy array, or the raw output
                                from the backend (the actual type may vary)
        """
        if Viewer.backend == 'gepetto-gui':
            if raw_data:
                raise ValueError(
                    "Raw data mode is not available using gepetto-gui.")
            if width is not None or height is not None:
                logger.warning("Cannot specify window size using gepetto-gui.")
            with tempfile.NamedTemporaryFile(suffix=".png") as f:  # Gepetto is not able to save the frame if the file does not have ".png" extension
                self.save_frame(f.name)  # It is not possible to capture frame directly using gepetto-gui
                img_obj = Image.open(f.name)
                rgb_array = np.array(img_obj)[:, :, :-1]
            return rgb_array
        else:
            # Send capture frame request to the background recorder process
            img_html = Viewer._backend_obj.capture_frame(width, height)

            # Parse the output to remove the html header, and
            # convert it into the desired output format.
            img_data = base64.decodebytes(str.encode(img_html[23:]))
            if raw_data:
                return img_data
            else:
                img_obj = Image.open(io.BytesIO(img_data))
                rgb_array = np.array(img_obj)
                return rgb_array

    @__must_be_open
    def save_frame(self,
                   image_path,
                   width=DEFAULT_CAPTURE_SIZE,
                   height=DEFAULT_CAPTURE_SIZE):
        """
        @brief      Save a snapshot in png format.

        @param[in]  image_path    Fullpath of the image (.png extension is mandatory for
                                  Gepetto-gui, it is .webp for Meshcat)
        @param[in]  width     Width for the image in pixels (not available with Gepetto-gui for now).
                              Optional: DEFAULT_CAPTURE_SIZE by default. None to keep unchanged.
        @param[in]  height    Height for the image in pixels (not available with Gepetto-gui for now).
                              Optional: DEFAULT_CAPTURE_SIZE by default. None to keep unchanged.
        """
        if Viewer.backend == 'gepetto-gui':
            image_path = str(pathlib.Path(image_path).with_suffix('.png'))
            self._client.captureFrame(self._window_id, image_path)
        else:
            img_data = self.capture_frame(width, height, raw_data=True)
            image_path = str(pathlib.Path(image_path).with_suffix('.webp'))
            with open(image_path, "wb") as f:
                f.write(img_data)

    @__must_be_open
    def refresh(self, wait=False):
        """
        @brief      Refresh the configuration of Robot in the viewer.

        @param[in]  wait    Whether or not to wait for rendering to finish.
        """
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
                self.__update_camera_transform()
                self._client.refresh()
            else:
                self.__updateGeometryPlacements(visual=True)
                for visual in self._rb.visual_model.geometryObjects:
                    T = self._rb.visual_data.oMg[\
                        self._rb.visual_model.getGeometryId(visual.name)].homogeneous
                    self._client.viewer[\
                        self.__getViewerNodeName(
                            visual, pin.GeometryType.VISUAL)].set_transform(T)
                self.__update_camera_transform()
            if wait:
                Viewer.wait()

    @__must_be_open
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
            self.__update_camera_transform()
            pin.framesForwardKinematics(self._rb.model, self._rb.data, q)  # This method is not called automatically by 'display' method
            if wait:
                Viewer.wait()

    def replay(self,
               evolution_robot,
               replay_speed,
               xyz_offset=None,
               wait=False):
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
            self.display(s.q, xyz_offset, wait)
            t_simu = (time.time() - init_time) * replay_speed
            i = bisect_right(t, t_simu)
            sleep(s.t - t_simu)
            wait = False  # It is enough to wait for the first timestep


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
                      record_video_path=None,
                      viewers=None,
                      start_paused=False,
                      wait_for_client=True,
                      travelling_frame=None,
                      camera_xyzrpy=None,
                      xyz_offset=None,
                      urdf_rgba=None,
                      backend=None,
                      window_name='python-pinocchio',
                      scene_name='world',
                      close_backend=None,
                      delete_robot_on_close=None,
                      verbose=True):
    """!
    @brief      Replay one or several robot trajectories in a viewer.

    @details    The ratio between the replay and the simulation time is kept constant to the desired ratio.
                One can choose between several backend (gepetto-gui or meshcat).

    @remark     The speed is independent of the plateform and the CPU power.

    @param[in]  trajectory_data     List of trajectory dictionary with keys:
                                    'evolution_robot': list of State object of increasing time
                                    'robot': jiminy robot (None if omitted)
                                    'use_theoretical_model': whether to use the theoretical or actual model
    @param[in]  mesh_root_path      Optional, path to the folder containing the URDF meshes.
    @param[in]  replay_speed        Speed ratio of the simulation.
                                    Optional: 1.0 by default
    @param[in]  record_video_path   Fullpath location where to save generated video (.mp4 extension is
                                    mandatory). Must be specified to enable video recording.
                                    Optional: None to disable. None by default.
    @param[in]  viewers             Already instantiated viewers, associated one by one in order to
                                    each trajectory data.
                                    Optional: None to disable. None by default.
    @param[in]  start_paused        Start the simulation is pause, waiting for keyboard input before
                                    starting to play the trajectories.
                                    Optional: False by default.
    @param[in]  wait_for_client     Wait for the client to finish loading the meshes before starting
                                    Optional: True by default.
    @param[in]  travelling_frame    Name of the frame to automatically follow with the camera.
                                    Optional: None to disable. None by default.
    @param[in]  camera_xyzrpy       Tuple (position [X, Y, Z], rotation [Roll, Pitch, Yaw]) corresponding
                                    to the absolute pose of the camera during replay, if travelling is
                                    disable, or the relative pose wrt the tracked frame otherwise.
                                    Optional: None to disable. None by default.
    @param[in]  xyz_offset          Constant translation of the root joint in world frame (1D numpy array).
                                    Optional: None to disable. None by default.
    @param[in]  urdf_rgba           RGBA code defining the color of the model. It is the same for each link.
                                    Optional: Original colors of each link. No alpha.
    @param[in]  backend             Backend, one of 'meshcat' or 'gepetto-gui'.
                                    Optional: If None, 'meshcat' is used in notebook environment and
                                    'gepetto-gui' otherwise. None by default.
    @param[in]  window_name         Name of the Gepetto-viewer's window in which to display the robot.
                                    Optional: Common default name if omitted.
    @param[in]  scene_name          Name of the Gepetto-viewer's scene in which to display the robot.
                                    Optional: Common default name if omitted.
    @param[in]  close_backend       Close backend automatically at exit.
                                    Optional: Enable by default if not (presumably) available beforehand.
    @param[in]  delete_robot_on_close    Whether or not to delete the robot from the viewer when closing it.
                                         Optional: True by default.
    @param[in]  verbose             Add information to keep track of the process.
                                    Optional: True by default.

    @return     The viewers used to play the trajectories.
    """
    if viewers is not None:
        # Make sure that viewers is a list
        if not isinstance(viewers, list):
            viewers = [viewers]

        # Make sure the viewers are still running if specified
        if not Viewer.is_open() is None:
            viewers = None
        for viewer in viewers:
            if not viewer.is_open():
                viewers = None
                break

        # Do not close backend by default if it was supposed to be available
        if close_backend is None:
            close_backend = False

    if viewers is None:
        # Delete robot by default only if not in notebook
        if delete_robot_on_close is None:
            delete_robot_on_close = not is_notebook()

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
                delete_robot_on_close=delete_robot_on_close,
                open_gui_if_parent=(record_video_path is None))
            viewers.append(viewer)

            # Close backend by default
            if close_backend is None:
                close_backend = True

    # Set camera pose or activate camera traveling if requested
    if travelling_frame is not None:
        if camera_xyzrpy is not None:
            viewers[0].attach_camera(travelling_frame, *camera_xyzrpy)
        else:
            viewers[0].attach_camera(travelling_frame)
    elif camera_xyzrpy is not None:
        viewers[0].set_camera_transform(*camera_xyzrpy)

    # Load robots in gepetto viewer
    if xyz_offset is None:
        xyz_offset = len(trajectory_data) * (None,)

    for i in range(len(trajectory_data)):
        try:
            viewers[i].display(
                trajectory_data[i]['evolution_robot'][0].q, xyz_offset[i])
        except Viewer._backend_exceptions:
            break

    # Wait for the meshes to finish loading if non video recording mode
    if record_video_path is None:
        if backend == 'meshcat':
            if verbose:
                print("Waiting for meshcat client in browser to connect: "\
                    f"{Viewer._backend_obj.gui.url()}")
            Viewer.wait(require_client=True)
            if verbose:
                print("Browser connected! Starting to replay the simulation.")

    # Handle start-in-pause mode
    if start_paused and not is_notebook():
        input("Press Enter to continue...")

    # Replay the trajectory
    if record_video_path is not None:
        # Extract and resample trajectory data at fixed framerate
        time_max = max([traj['evolution_robot'][-1].t
                        for traj in trajectory_data])
        time_evolution = np.arange(
            0.0, time_max, replay_speed / VIDEO_FRAMERATE)
        position_evolution = []
        for j in range(len(trajectory_data)):
            data_orig = trajectory_data[j]['evolution_robot']
            t_orig = np.array([s.t for s in data_orig])
            pos_orig = np.stack([s.q for s in data_orig], axis=0)
            pos_interp = interp1d(
                t_orig, pos_orig,
                kind='linear', bounds_error=False,
                fill_value=(pos_orig[0], pos_orig[-1]), axis=0)
            position_evolution.append(pos_interp(time_evolution))

        # Play trajectories without multithreading and record_video
        for i in tqdm(range(len(time_evolution)),
                      desc="Rendering frames",
                      disable=(not verbose)):
            for j in range(len(trajectory_data)):
                viewers[j].display(
                    position_evolution[j][i], xyz_offset=xyz_offset[j])
            if Viewer.backend != 'meshcat':
                import cv2
                frame = viewers[0].capture_frame(VIDEO_SIZE[1], VIDEO_SIZE[0])
                if i == 0:
                    record_video_path = str(
                        pathlib.Path(record_video_path).with_suffix('.mp4'))
                    out = cv2.VideoWriter(
                        record_video_path, cv2.VideoWriter_fourcc(*'vp09'),
                        fps=VIDEO_FRAMERATE, frameSize=frame.shape[1::-1])
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                if i == 0:
                    record_video_path = str(
                        pathlib.Path(record_video_path).with_suffix('.webm'))
                    viewers[0]._backend_obj.start_recording(
                        VIDEO_FRAMERATE, *VIDEO_SIZE)
                viewers[0]._backend_obj.add_frame()
        if Viewer.backend != 'meshcat':
            out.release()
        else:
            viewers[0]._backend_obj.stop_recording(record_video_path)
    else:
        def replay_thread(viewer, *args, **kwargs):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            viewer.replay(*args, **kwargs)

        # Play trajectories with multithreading
        threads = []
        for i in range(len(trajectory_data)):
            threads.append(Thread(
                target=replay_thread,
                args=(viewers[i],
                      trajectory_data[i]['evolution_robot'],
                      replay_speed,
                      xyz_offset[i],
                      True)))
        for i in range(len(trajectory_data)):
            threads[i].daemon = True
            threads[i].start()
        for i in range(len(trajectory_data)):
            threads[i].join()

    # Disable camera traveling it was enabled
    if travelling_frame is not None:
        viewers[0].detach_camera()

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
