import os
import re
import io
import time
import types
import psutil
import shutil
import base64
import atexit
import logging
import pathlib
import asyncio
import tempfile
import subprocess
import webbrowser
import numpy as np
import multiprocessing
from tqdm import tqdm
from PIL import Image
from functools import wraps
from bisect import bisect_right
from threading import Thread, Lock
from scipy.interpolate import interp1d
from typing import Optional, Union, List, Tuple, Dict, Any

import zmq
import meshcat.transformations as mtf

import pinocchio as pin
from pinocchio import SE3, se3ToXYZQUAT, XYZQUATToSe3
from pinocchio.rpy import rpyToMatrix, matrixToRpy
from pinocchio.visualize import GepettoVisualizer

from . import core as jiminy
from .state import State
from .meshcat.utilities import is_notebook
from .meshcat.wrapper import MeshcatWrapper
from .meshcat.meshcat_visualizer import MeshcatVisualizer


CAMERA_INV_TRANSFORM_MESHCAT = rpyToMatrix(np.array([-np.pi / 2, 0.0, 0.0]))
DEFAULT_CAMERA_XYZRPY = ([7.5, 0.0, 1.4], [1.4, 0.0, np.pi / 2])
DEFAULT_CAPTURE_SIZE = 500
VIDEO_FRAMERATE = 30
VIDEO_SIZE = (1000, 1000)


# Determine set the of available backends
backends_available = {'meshcat': MeshcatVisualizer}
if __import__('platform').system() == 'Linux':
    import importlib
    if (importlib.util.find_spec("gepetto") is not None and
            importlib.util.find_spec("omniORB") is not None):
        backends_available['gepetto-gui'] = GepettoVisualizer


def _default_backend() -> str:
    """Determine the default backend viewer, depending on the running
    environment and the set of available backends.
    """
    if is_notebook() or 'gepetto-gui' not in backends_available:
        return 'meshcat'
    else:
        return 'gepetto-gui'


def _get_backend_exceptions(backend: Optional[str] = None) -> List[Exception]:
    if backend is None:
        backend = _default_backend()
    if backend.startswith('gepetto'):
        import gepetto
        import omniORB
        return (omniORB.CORBA.COMM_FAILURE,
                omniORB.CORBA.TRANSIENT,
                gepetto.corbaserver.gepetto.Error)
    else:
        return (zmq.error.Again, zmq.error.ZMQError)


# Create logger
class _DuplicateFilter:
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv


logger = logging.getLogger(__name__)
logger.addFilter(_DuplicateFilter())


def sleep(dt: float) -> None:
    """Function to provide cross-plateform time sleep with maximum accuracy.

    .. warning::
        Use this method with cautious since it relies on busy looping principle
        instead of system scheduler. As a result, it wastes a lot more
        resources than time.sleep. However, it is the only way to ensure
        accurate delay on a non-real-time systems such as Windows 10.

    :param dt: Sleep duration in seconds.
    """
    _ = time.perf_counter() + dt
    while time.perf_counter() < _:
        pass


class _ProcessWrapper:
    """Wrap `multiprocessing.Process`, `subprocess.Popen`, and `psutil.Process`
    in the same object to have the same user interface.

    It also makes sure that the process is properly terminated at Python exits,
    and without zombies left behind.
    """
    def __init__(self,
                 proc: Union[multiprocessing.Process,
                             subprocess.Popen, psutil.Process],
                 kill_at_exit: bool = False):
        self._proc = proc
        # Make sure the process is killed at Python exit
        if kill_at_exit:
            atexit.register(self.kill)

    def is_parent(self) -> bool:
        return isinstance(self._proc, (
            subprocess.Popen, multiprocessing.Process))

    def is_alive(self) -> bool:
        if isinstance(self._proc, subprocess.Popen):
            return self._proc.poll() is None
        elif isinstance(self._proc, multiprocessing.Process):
            return self._proc.is_alive()
        elif isinstance(self._proc, psutil.Process):
            try:
                return self._proc.status() in [
                    psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]
            except psutil.NoSuchProcess:
                return False

    def wait(self, timeout: Optional[float] = None) -> bool:
        if isinstance(self._proc, multiprocessing.Process):
            return self._proc.join(timeout)
        elif isinstance(self._proc, (
                subprocess.Popen, psutil.Process)):
            return self._proc.wait(timeout)

    def kill(self) -> None:
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
    backend = _default_backend()
    _backend_obj = None
    _backend_exceptions = _get_backend_exceptions()
    _backend_proc = None
    _backend_robot_names = set()
    _lock = Lock()  # Unique lock for every viewer in same thread by default

    def __init__(self,
                 robot: jiminy.Robot,
                 use_theoretical_model: bool = False,
                 urdf_rgba: Optional[Tuple[float, float, float, float]] = None,
                 lock: Optional[Lock] = None,
                 backend: Optional[str] = None,
                 open_gui_if_parent: bool = True,
                 delete_robot_on_close: bool = False,
                 robot_name: Optional[str] = None,
                 window_name: str = 'jiminy',
                 scene_name: str = 'world'):
        """
        :param robot: Jiminy.Robot to display.
        :param use_theoretical_model: Whether to use the theoretical (rigid)
                                      model or the actual (flexible) model of
                                      this robot.
        :param urdf_rgba: RGBA color to use to display this robot, as a list
                          of 4 floating-point values between 0.0 and 1.0.
                          Optional: It will override the original color of the
                          meshes if specified.
        :param lock: Custom threading.Lock. Required for parallel rendering.
                     It is required since some backends does not support
                     multiple simultaneous connections (e.g. corbasever).
                     Optional: Unique lock of the current thread by default.
        :param backend: The name of the desired backend to use for rendering.
                        It can be either 'gepetto-gui' or 'meshcat'.
                        Optional: 'gepetto-gui' by default if available and not
                        running from a notebook, 'meshcat' otherwise.
        :param open_gui_if_parent: Open GUI if new viewer's backend server is
                                   started.
        :param delete_robot_on_close: Enable automatic deletion of the robot
                                      when closing.
        :param robot_name: Unique robot name, to identify each robot.
                           Optional: Randomly generated identifier by default.
        :param window_name: Window name, used only when gepetto-gui is used
                            as backend. Note that it is not allowed to be equal
                            to the window name.
        :param scene_name: Scene name, used only with gepetto-gui backend.
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
        # without explicitly calling `set_camera_transform`.
        self.detach_camera()

        # Make sure that the windows, scene and robot names are valid
        if scene_name == window_name:
            raise ValueError(
                "The name of the scene and window must be different.")

        if robot_name in Viewer._backend_robot_names:
            raise ValueError(
                "Robot name already exists but must be unique. Please choose "
                "a different one, or close the associated viewer.")

        # Select the desired backend
        if backend is None:
            backend = Viewer.backend
        else:
            backend = backend.lower()  # Make sure backend's name is lowercase
            if backend not in backends_available:
                raise ValueError("%s backend not available." % backend)

        # Update the backend currently running, if any
        if Viewer.backend != backend and Viewer._backend_obj is not None:
            Viewer.close()
            logging.warning("Different backend already running. Closing it...")
        Viewer.backend = backend

        # Configure exception handling
        Viewer._backend_exceptions = _get_backend_exceptions(backend)

        # Check if the backend is still available, if any
        if Viewer._backend_obj is not None:
            is_backend_running = True
            if not Viewer.is_open():
                is_backend_running = False
            if Viewer.backend.startswith('gepetto'):
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

        # Create a unique temporary directory, specific to this viewer instance
        self._tempdir = tempfile.mkdtemp(
            prefix="_".join((window_name, scene_name, robot_name, "")))

        # Extract the right Pinocchio model
        if self.use_theoretical_model:
            pinocchio_model = robot.pinocchio_model_th
            pinocchio_data = robot.pinocchio_data_th
        else:
            pinocchio_model = robot.pinocchio_model
            pinocchio_data = robot.pinocchio_data

        # Generate colorized URDF file if using gepetto-gui backend, since it
        # is not supported by default, because of memory optimizations.
        if Viewer.backend.startswith('gepetto'):
            if urdf_rgba is not None:
                alpha = urdf_rgba[3]
                self.urdf_path = Viewer._get_colorized_urdf(
                    self.urdf_path, urdf_rgba[:3], self._tempdir)
            else:
                alpha = 1.0

        # Create robot visual model
        visual_model = pin.buildGeomFromUrdf(
            pinocchio_model, self.urdf_path, robot.mesh_package_dirs,
            pin.GeometryType.VISUAL)

        # Access the current backend or create one if none is available
        self.__is_open = False
        self.is_backend_parent = False
        try:
            # Create viewer backend if necessary
            if Viewer._backend_obj is None:
                Viewer._backend_obj, Viewer._backend_proc = \
                    Viewer.__get_client(start_if_needed=True)
                self.is_backend_parent = Viewer._backend_proc.is_parent()
            self._gui = Viewer._backend_obj.gui
            self.__is_open = True

            # Create backend wrapper to get (almost) backend-independent API
            self._client = backends_available[Viewer.backend](
                pinocchio_model, robot.collision_model, visual_model)
            self._client.data = pinocchio_data
            self._client.collision_data = robot.collision_data

            # Create the scene and load robot
            if Viewer.backend.startswith('gepetto'):
                # Initialize the viewer
                self._client.initViewer(
                    viewer=Viewer._backend_obj, windowName=window_name,
                    sceneName=scene_name, loadModel=False)

                # Add missing scene elements
                if Viewer.backend.startswith('gepetto'):
                    self._gui.addFloor('/'.join((scene_name, "floor")))
                    self._gui.addLandmark(scene_name, 0.1)

                # Load the robot
                self._client.loadViewerModel(rootNodeName=self.robot_name)
                robot_node_path = '/'.join((scene_name, self.robot_name))
                try:
                    self._gui.setFloatProperty(robot_node_path, 'Alpha', alpha)
                except Viewer._backend_exceptions:
                    # Old Gepetto versions do no have 'Alpha' attribute but
                    # 'Transparency'.
                    self._gui.setFloatProperty(
                        robot_node_path, 'Transparency', 1 - alpha)
            else:
                # Initialize the viewer
                self._client.initViewer(
                    viewer=self._gui, must_open=False, loadModel=False)

                # Load the robot
                root_name = '/'.join((self.scene_name, self.robot_name))
                self._client.loadViewerModel(
                    rootNodeName=root_name, color=urdf_rgba)

            Viewer._backend_robot_names.add(self.robot_name)

            # Open a gui window in browser, since the server is headless.
            # Note that the scene is created automatically as client level, it
            # is not managed by the server.
            if Viewer.backend.startswith('meshcat'):
                if not is_notebook():
                    if self.is_backend_parent and open_gui_if_parent:
                        self.open_gui()
                else:
                    # Opening a new display cell automatically if not backend
                    # parent, or if there is no display cell already opened.
                    # Indeed, the user is probably expecting a display cell to
                    # open in such cases, but there is no fixed rule.
                    if not Viewer._backend_proc.is_parent() or \
                            Viewer._backend_obj.comm_manager.n_comm == 0:
                        self.open_gui()
        except Exception as e:
            raise RuntimeError(
                "Impossible to create or connect to backend.") from e

        # Set default camera pose
        if self.is_backend_parent:
            self.set_camera_transform()

        # Refresh the viewer since the positions of the meshes and their
        # visibility mode are not properly set at this point.
        self.refresh()

    def __del__(self):
        """Destructor.

        .. note::
            It automatically close the viewer before being garbage collected.
        """
        self.close()

    def __must_be_open(fct):
        @wraps(fct)
        def fct_safe(*args, **kwargs):
            self = None
            if args and isinstance(args[0], Viewer):
                self = args[0]
            self = kwargs.get('self', self)
            if not Viewer.is_open(self):
                raise RuntimeError(
                    "No backend available. Please start one before calling "
                    f"'{fct.__name__}'.")
            return fct(*args, **kwargs)
        return fct_safe

    @staticmethod
    def open_gui(start_if_needed: bool = False) -> bool:
        """Open a new viewer graphical interface.

        .. note::
            This method is not supported by Gepetto-gui since it does not have
            a classical server/client mechanism. One and only one graphical
            interface (client) can be opened, and its lifetime is tied to the
            one of the server itself.
        """
        if Viewer.backend.startswith('gepetto'):
            raise RuntimeError(
                "Showing client is only available using 'meshcat' backend.")
        else:
            if Viewer._backend_obj is None:
                Viewer._backend_obj, Viewer._backend_proc = \
                    Viewer.__get_client(start_if_needed)
            viewer_url = Viewer._backend_obj.gui.url()

            if is_notebook():
                import urllib
                from IPython.core.display import HTML, display

                # Scrap the viewer html content, including javascript
                # dependencies
                html_content = urllib.request.urlopen(
                    viewer_url).read().decode()
                pattern = '<script type="text/javascript" src="%s"></script>'
                scripts_js = re.findall(pattern % '(.*)', html_content)
                for file in scripts_js:
                    file_path = os.path.join(viewer_url, file)
                    js_content = urllib.request.urlopen(
                        file_path).read().decode()
                    html_content = html_content.replace(pattern % file, f"""
                    <script type="text/javascript">
                    {js_content}
                    </script>""")

                if is_notebook() == 1:
                    # Embed HTML in iframe on Jupyter, since it is not
                    # possible to load HTML/Javascript content directly.
                    html_content = html_content.replace(
                        "\"", "&quot;").replace("'", "&apos;")
                    display(HTML(f"""
                        <div class="resizable" style="
                                height: 400px; width: 100%;
                                overflow-x: auto; overflow-y: hidden;
                                resize: both">
                            <iframe srcdoc="{html_content}" style="
                                width: 100%; height: 100%; border: none;">
                            </iframe>
                        </div>
                    """))
                else:
                    # Adjust the initial window size
                    html_content = html_content.replace(
                        '<div id="meshcat-pane">', """
                        <div id="meshcat-pane" class="resizable" style="
                                height: 400px; width: 100%;
                                overflow-x: auto; overflow-y: hidden;
                                resize: both">
                    """)
                    display(HTML(html_content))
            else:
                try:
                    webbrowser.get()
                    webbrowser.open(viewer_url, new=2, autoraise=True)
                except webbrowser.Error:  # Fail if not browser is available
                    logger.warning(
                        "No browser available for display. "
                        "Please install one manually.")
                    return  # Skip waiting since there is nothing to wait for

            # Wait for the display to finish loading
            Viewer.wait(require_client=True)

    @staticmethod
    @__must_be_open
    def wait(require_client: bool = False) -> None:
        """Wait for all the meshes to finish loading in every clients.

        :param require_client: Wait for at least one client to be available
                               before checking for mesh loading.
        """
        if Viewer.backend != 'gepetto-gui':
            # Gepetto-gui is synchronous, so it cannot not be already loaded.
            Viewer._backend_obj.wait(require_client)

    @staticmethod
    def is_alive() -> bool:
        """Check if the backend server is running and responding to queries.
        """
        return Viewer._backend_proc is not None and \
            Viewer._backend_proc.is_alive()

    def is_open(self=None) -> bool:
        """Check if a given viewer instance is open, or if the backend server
        is running if no instance is specified.
        """
        is_open_ = Viewer.is_alive()
        if self is not None:
            is_open_ = is_open_ and self.__is_open
        return is_open_

    def close(self=None) -> None:
        """Close a given viewer instance, or all of them if no instance is
        specified.

        .. note::
            Calling this method with an viewer instance always closes the
            client. It may also remove the robot from the server if the viewer
            attribute `delete_robot_on_close` is True. Moreover, it is the one
            having started the backend server, it also terminates it, resulting
            in closing every viewer somehow. It results in the same outcome
            than calling this method without specifying any viewer instance.
        """
        try:
            if Viewer.backend == 'meshcat' and Viewer._backend_obj is not None:
                Viewer._backend_obj.gui.window.zmq_socket.RCVTIMEO = 50
            if self is None:
                self = Viewer
            else:
                # Consider that the robot name is now available, no matter
                # whether the robot has actually been deleted or not.
                Viewer._backend_robot_names.discard(self.robot_name)
                if self.delete_robot_on_close:
                    # In case 'close' is called twice.
                    self.delete_robot_on_close = False
                    Viewer._delete_nodes_viewer(
                        ['/'.join((self.scene_name, self.robot_name))])
            if self == Viewer:
                # NEVER closing backend if closing instances, even for the
                # parent. It will be closed at Python exit automatically.
                Viewer._backend_robot_names.clear()
                if Viewer.backend == 'meshcat' and \
                        Viewer._backend_obj is not None:
                    Viewer._backend_obj.close()
                    _ProcessWrapper(Viewer._backend_obj.recorder.proc).kill()
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
        except Exception:  # This method must not fail under any circumstances
            pass

    @staticmethod
    def _get_colorized_urdf(urdf_path: str,
                            rgb: List[float],
                            output_root_path: Optional[str] = None) -> str:
        """Generate a unique colorized URDF.

        .. note::
            Multiple identical URDF model of different colors can be loaded in
            Gepetto-viewer this way.

        :param urdf_path: Full path of the URDF file.
        :param rgb: RGB code defining the color of the model. It is the same
                    for each link.
        :param output_root_path: Root directory of the colorized URDF data.
                                 Optional: temporary directory by default.

        :returns: Full path of the colorized URDF file.
        """
        # Convert RGB array to string and xml tag. Don't close tag with '>',
        # in order to handle <color/> and <color></color>.
        color_string = "%.3f_%.3f_%.3f_1.0" % tuple(rgb)
        color_tag = "<color rgba=\"%.3f %.3f %.3f 1.0\"" % tuple(rgb)

        # Create the output directory
        if output_root_path is None:
            output_root_path = tempfile.mkdtemp()
        colorized_data_dir = os.path.join(
            output_root_path, "colorized_urdf_rgba_" + color_string)
        os.makedirs(colorized_data_dir, exist_ok=True)
        colorized_urdf_path = os.path.join(
            colorized_data_dir, os.path.basename(urdf_path))

        # Copy the meshes in temporary directory and update paths in URDF file
        with open(urdf_path, 'r') as urdf_file:
            colorized_contents = urdf_file.read()

        for mesh_fullpath in re.findall(
                '<mesh filename="(.*)"', colorized_contents):
            colorized_mesh_fullpath = os.path.join(
                colorized_data_dir, mesh_fullpath[1:])
            colorized_mesh_path = os.path.dirname(colorized_mesh_fullpath)
            if not os.access(colorized_mesh_path, os.F_OK):
                os.makedirs(colorized_mesh_path)
            shutil.copy2(mesh_fullpath, colorized_mesh_fullpath)
            colorized_contents = colorized_contents.replace(
                '"' + mesh_fullpath + '"',
                '"' + colorized_mesh_fullpath + '"', 1)
        colorized_contents = re.sub(
            r'<color rgba="[\d. ]*"', color_tag, colorized_contents)

        with open(colorized_urdf_path, 'w') as f:
            f.write(colorized_contents)

        return colorized_urdf_path

    @staticmethod
    def __get_client(
        start_if_needed: bool = False,
        close_at_exit: bool = True,
        timeout: int = 2000) -> Tuple[
            Optional[Union['gepetto.corbaserver.client', MeshcatWrapper]],  # noqa
            Optional[_ProcessWrapper]]:
        """Get a pointer to the running process of Gepetto-Viewer.

        This method can be used to open a new process if necessary.

        :param start_if_needed: Whether a new process must be created if no
                                running process is found.
                                Optional: False by default
        :param timeout: Wait some millisecond before considering starting new
                        server has failed.
                        Optional: 1s by default
        :param close_at_exit: Terminate backend server at Python exit.
                              Optional: True by default

        :returns: Pointer to the running Gepetto-viewer Client and its PID.
        """
        if Viewer.backend.startswith('gepetto'):
            from gepetto.corbaserver.client import Client as gepetto_client

            def _gepetto_client_connect(get_proc_info=False):
                nonlocal close_at_exit

                # Get the existing Gepetto client
                client = gepetto_client()

                # Try to fetch the list of scenes to make sure that the Gepetto
                # client is responding.
                client.gui.getSceneList()

                # Get the associated process information if requested
                if not get_proc_info:
                    return client
                proc = [p for p in psutil.process_iter()
                        if p.cmdline() and 'gepetto-gui' in p.cmdline()[0]][0]
                return client, _ProcessWrapper(proc, close_at_exit)

            try:
                return _gepetto_client_connect(get_proc_info=True)
            except Viewer._backend_exceptions:
                try:
                    return _gepetto_client_connect(get_proc_info=True)
                except Viewer._backend_exceptions:
                    if start_if_needed:
                        FNULL = open(os.devnull, 'w')
                        proc = subprocess.Popen(
                            ['gepetto-gui'], shell=False, stdout=FNULL,
                            stderr=FNULL)
                        proc = _ProcessWrapper(proc, close_at_exit)
                        # Must try at least twice for robustness
                        for _ in range(max(2, int(timeout / 200))):
                            time.sleep(0.2)
                            try:
                                return _gepetto_client_connect(), proc
                            except Viewer._backend_exceptions:
                                pass
                        raise RuntimeError(
                            "Impossible to open Gepetto-viewer.")
            return None, None
        else:
            # List of connections likely to correspond to Meshcat servers
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
            proc = _ProcessWrapper(proc, close_at_exit)

            return client, proc

    @staticmethod
    @__must_be_open
    def _delete_nodes_viewer(nodes_path: str) -> None:
        """Delete a 'node' in Gepetto-viewer.

        .. note::
            Be careful, one must specify the full path of a node, including all
            parent group, but without the window name, ie
            'scene_name/robot_name' to delete the robot.

        :param nodes_path: Full path of the node to delete
        """
        try:
            if Viewer.backend.startswith('gepetto'):
                for node_path in nodes_path:
                    if node_path in Viewer._backend_obj.gui.getNodeList():
                        Viewer._backend_obj.gui.deleteNode(node_path, True)
            else:
                for node_path in nodes_path:
                    Viewer._backend_obj.gui[node_path].delete()
        except Viewer._backend_exceptions:
            pass

    @__must_be_open
    def set_camera_transform(self,
                             translation: Union[Tuple[float, float, float],
                                                np.ndarray] = None,
                             rotation: Union[Tuple[float, float, float],
                                             np.ndarray] = None,
                             relative: Optional[str] = None) -> None:
        """Apply transform to the camera pose.

        :param translation: Position [X, Y, Z] as a list or 1D array
        :param rotation: Rotation [Roll, Pitch, Yaw] as a list or 1D np.array
        :param relative:
            .. raw:: html

                How to apply the transform:

            - **None:** absolute
            - **'camera':** relative to the current camera pose
            - **other string:** relative to a robot frame, not accounting for
              the rotation (travelling)
        """
        # Handling of translation and rotation arguments
        if relative is not None and relative != 'camera':
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
            if Viewer.backend.startswith('gepetto'):
                H_orig = XYZQUATToSe3(
                    self._gui.getCameraTransform(self._client.windowID))
            else:
                raise RuntimeError(
                    "relative='camera' option is not available in Meshcat.")
        elif relative is not None:
            # Get the body position, not taking into account the rotation
            body_id = self.pinocchio_model.getFrameId(relative)
            try:
                body_transform = self.pinocchio_data.oMf[body_id]
            except IndexError:
                raise ValueError("'relative' set to non existing frame.")
            H_orig = SE3(np.eye(3), body_transform.translation)

        # Perform the desired rotation
        if Viewer.backend.startswith('gepetto'):
            H_abs = SE3(rotation_mat, translation)
            if relative is None:
                self._gui.setCameraTransform(
                    self._client.windowID, se3ToXYZQUAT(H_abs).tolist())
            else:
                # Not using recursive call for efficiency
                H_abs = H_orig * H_abs
                self._gui.setCameraTransform(
                    self._client.windowID, se3ToXYZQUAT(H_abs).tolist())
        elif Viewer.backend.startswith('meshcat'):
            if relative is None:
                # Meshcat camera is rotated by -pi/2 along Roll axis wrt the
                # usual convention in robotics.
                translation = CAMERA_INV_TRANSFORM_MESHCAT @ translation
                rotation = matrixToRpy(
                    CAMERA_INV_TRANSFORM_MESHCAT @ rotation_mat)
                self._gui["/Cameras/default/rotated/<object>"].\
                    set_transform(mtf.compose_matrix(
                        translate=translation, angles=rotation))
            else:
                H_abs = SE3(rotation_mat, translation)
                H_abs = H_orig * H_abs
                # Note that the original rotation is not modified.
                self.set_camera_transform(H_abs.translation, rotation)

    def attach_camera(self,
                      frame: Optional[str] = None,
                      translation: Optional[Union[Tuple[float, float, float],
                                                  np.ndarray]] = None,
                      rotation: Optional[Union[Tuple[float, float, float],
                                               np.ndarray]] = None) -> None:
        """Attach the camera to a given robot frame.

        Only the position of the frame is taken into account. A custom relative
        pose of the camera wrt to the frame can be further specified.

        :param frame: Frame of the robot to follow with the camera.
        :param translation: Relative position [X, Y, Z] of the camera wrt the
                            frame.
        :param rotation: Relative rotation [Roll, Pitch, Yaw] of the camera wrt
                         the frame.
        """
        def __update_camera_transform(self):
            nonlocal frame, translation, rotation
            self.set_camera_transform(translation, rotation, relative=frame)
        self.__update_camera_transform = types.MethodType(
            __update_camera_transform, self)

    def detach_camera(self) -> None:
        """Detach the camera.

        Must be called to undo `attach_camera`, so that it will stop
        automatically tracking a frame.
        """
        self.__update_camera_transform = lambda: None

    @__must_be_open
    def capture_frame(self,
                      width: int = DEFAULT_CAPTURE_SIZE,
                      height: int = DEFAULT_CAPTURE_SIZE,
                      raw_data: bool = False) -> Union[np.ndarray, str]:
        """Take a snapshot and return associated data.

        :param width: Width for the image in pixels (not available with
                      Gepetto-gui for now). None to keep unchanged.
                      Optional: DEFAULT_CAPTURE_SIZE by default.
        :param height: Height for the image in pixels (not available with
                       Gepetto-gui for now). None to keep unchanged.
                       Optional: DEFAULT_CAPTURE_SIZE by default.
        :param raw_data: Whether to return a 2D numpy array, or the raw output
                         from the backend (the actual type may vary).
        """
        if Viewer.backend.startswith('gepetto'):
            if raw_data:
                raise ValueError(
                    "Raw data mode is not available using gepetto-gui.")
            if width is not None or height is not None:
                logger.warning("Cannot specify window size using gepetto-gui.")
            # It is not possible to capture frame directly using gepetto-gui,
            # and it is not able to save the frame if the file does not have
            # ".png" extension.
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                self.save_frame(f.name)
                img_obj = Image.open(f.name)
                rgb_array = np.array(img_obj)[:, :, :-1]
            return rgb_array
        else:
            # Send capture frame request to the background recorder process
            img_html = Viewer._backend_obj.capture_frame(width, height)

            # Parse the output to remove the html header, and convert it into
            # the desired output format.
            img_data = base64.decodebytes(str.encode(img_html[23:]))
            if raw_data:
                return img_data
            else:
                img_obj = Image.open(io.BytesIO(img_data))
                rgb_array = np.array(img_obj)
                return rgb_array

    @__must_be_open
    def save_frame(self,
                   image_path: str,
                   width: int = DEFAULT_CAPTURE_SIZE,
                   height: int = DEFAULT_CAPTURE_SIZE) -> None:
        """Save a snapshot in png format.

        :param image_path: Fullpath of the image (.png extension is mandatory
                           for Gepetto-gui, it is .webp for Meshcat)
        :param width: Width for the image in pixels (not available with
                      Gepetto-gui for now). None to keep unchanged.
                      Optional: DEFAULT_CAPTURE_SIZE by default.
        :param height: Height for the image in pixels (not available with
                       Gepetto-gui for now). None to keep unchanged.
                       Optional: DEFAULT_CAPTURE_SIZE by default.
        """
        if Viewer.backend.startswith('gepetto'):
            image_path = str(pathlib.Path(image_path).with_suffix('.png'))
            self._gui.captureFrame(self._client.windowID, image_path)
        else:
            img_data = self.capture_frame(width, height, raw_data=True)
            image_path = str(pathlib.Path(image_path).with_suffix('.webp'))
            with open(image_path, "wb") as f:
                f.write(img_data)

    @__must_be_open
    def display_visuals(self, visibility: bool):
        """Set the visibility of the visual model of the robot.

        :param visibility: Whether to enable or disable display of the visual
                           model.
        """
        self._client.displayVisuals(visibility)
        self.refresh()

    @__must_be_open
    def display_collisions(self, visibility: bool):
        """Set the visibility of the collision model of the robot.

        :param visibility: Whether to enable or disable display of the visual
                           model.
        """
        self._client.displayCollisions(visibility)
        self.refresh()

    @__must_be_open
    def refresh(self,
                force_update_visual: bool = False,
                force_update_collision: bool = False,
                wait: bool = False) -> None:
        """Refresh the configuration of Robot in the viewer.

        This method is also in charge of updating the camera placement for
        traveling.

        .. note::
            This method is copy-pasted from `Pinocchio.visualize.*.display`
            method, after removing parts responsible of update pinocchio
            data and collision data. Visual data must still be updated.

        :param force_update_visual: Force update of visual geometries.
        :param force_update_collision: Force update of collision geometries.
        :param wait: Whether or not to wait for rendering to finish.
        """
        with self._lock:
            # Render both visual and collision geometries
            model_list, data_list, model_type_list = [], [], []
            if self._client.display_collisions or force_update_collision:
                model_list.append(self._client.collision_model)
                data_list.append(self._client.collision_data)
                model_type_list.append(pin.GeometryType.COLLISION)
            if self._client.display_visuals or force_update_visual:
                model_list.append(self._client.visual_model)
                data_list.append(self._client.visual_data)
                model_type_list.append(pin.GeometryType.VISUAL)

            for model, data, in zip(model_list, data_list):
                pin.updateGeometryPlacements(
                    self._client.model, self._client.data, model, data)

            if Viewer.backend.startswith('gepetto'):
                for model, data, model_type in zip(
                        model_list, data_list, model_type_list):
                    self._gui.applyConfigurations(
                        [self._client.getViewerNodeName(geometry, model_type)
                            for geometry in model.geometryObjects],
                        [pin.se3ToXYZQUATtuple(data.oMg[
                            model.getGeometryId(geometry.name)])
                            for geometry in model.geometryObjects])
            else:
                for model, data, model_type in zip(
                        model_list, data_list, model_type_list):
                    for i, geom in enumerate(model.geometryObjects):
                        M = data.oMg[i]
                        S = np.diag(np.concatenate((
                            geom.meshScale, np.array([1.0]))).flat)
                        T = M.homogeneous.dot(S)
                        nodeName = self._client.getViewerNodeName(
                            geom, model_type)
                        self._client.viewer[nodeName].set_transform(T)

            # Update the camera placement
            self.__update_camera_transform()

            # Refreshing viewer backend manually is necessary for gepetto-gui
            if Viewer.backend.startswith('gepetto'):
                self._gui.refresh()

            # Wait for the backend viewer to finish rendering if requested
            if wait:
                Viewer.wait()

    @__must_be_open
    def display(self,
                q: np.ndarray,
                xyz_offset: Optional[np.ndarray] = None,
                wait: bool = False) -> None:
        """Update the configuration of the robot.

        .. warning::
            It will alter original robot data if viewer attribute
            `use_theoretical_model` is false.

        :param q: Configuration of the robot.
        :param xyz_offset: Freeflyer position offset. Note that it does not
                           check for the robot actually have a freeflyer.
        :param wait: Whether or not to wait for rendering to finish.
        """
        assert self._client.model.nq == q.shape[0], (
            "The configuration vector does not have the right size.")

        # Apply offset on the freeflyer, if requested.
        # Note that it is NOT checking that the robot has a freeflyer.
        if xyz_offset is not None:
            q = q.copy()  # Make a copy to avoid altering the original data
            q[:3] += xyz_offset

        # Update pinocchio and collision data
        pin.forwardKinematics(self._client.model, self._client.data, q)
        pin.updateGeometryPlacements(
            self._client.model, self._client.data,
            self._client.collision_model, self._client.collision_data)
        pin.framesForwardKinematics(self._client.model, self._client.data, q)

        # Refresh the viewer
        self.refresh(wait)

    def replay(self,
               evolution_robot: List[State],
               replay_speed: float,
               xyz_offset: Optional[np.ndarray] = None,
               wait: bool = False) -> None:
        """Replay a complete robot trajectory at a given real-time ratio.

        .. warning::
            It will alter original robot data if viewer attribute
            `use_theoretical_model` is false.

        :param evolution_robot: List of State object of increasing time
        :param replay_speed: Real-time ratio
        :param xyz_offset: Freeflyer position offset. Note that it does not
                           check for the robot actually have a freeflyer.
        :param wait: Whether or not to wait for rendering to finish.
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


def extract_viewer_data_from_log(log_data: Dict[str, np.ndarray],
                                 robot: jiminy.Robot) -> Dict[str, Any]:
    """Extract the minimal required information from raw log data in order to
    replay the simulation in a viewer.

    It extracts only the required data for replay, namely the evolution over
    time of the joints positions.

    :param log_data: Data from the log file, in a dictionnary.
    :param robot: Jiminy robot.

    :returns: Trajectory dictionary. The actual trajectory corresponds to the
              field "evolution_robot" and it is a list of State object. The
              other fields are additional information.
    """

    # Get the current robot model options
    model_options = robot.get_model_options()

    # Extract the joint positions time evolution
    t = log_data["Global.Time"]
    try:
        qe = np.stack([log_data["HighLevelController." + s]
                       for s in robot.logfile_position_headers], axis=-1)
    except KeyError:
        model_options['dynamics']['enableFlexibleModel'] = \
            not robot.is_flexible
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
    for t_i, q_i in zip(t, qe):
        evolution_robot.append(State(t=t_i, q=q_i))

    return {'evolution_robot': evolution_robot,
            'robot': robot,
            'use_theoretical_model': use_theoretical_model}


def play_trajectories(trajectory_data: Dict[str, Any],
                      replay_speed: float = 1.0,
                      record_video_path: Optional[str] = None,
                      viewers: List[Viewer] = None,
                      start_paused: bool = False,
                      wait_for_client: bool = True,
                      travelling_frame: Optional[str] = None,
                      camera_xyzrpy: Optional[Tuple[
                          Union[Tuple[float, float, float], np.ndarray],
                          Union[Tuple[float, float, float],
                                np.ndarray]]] = None,
                      xyz_offset: Optional[List[Union[
                          Tuple[float, float, float], np.ndarray]]] = None,
                      urdf_rgba: Optional[List[
                          Tuple[float, float, float, float]]] = None,
                      backend: Optional[str] = None,
                      window_name: str = 'jiminy',
                      scene_name: str = 'world',
                      close_backend: Optional[bool] = None,
                      delete_robot_on_close: Optional[bool] = None,
                      verbose: bool = True) -> List[Viewer]:
    """Replay one or several robot trajectories in a viewer.

    The ratio between the replay and the simulation time is kept constant to
    the desired ratio. One can choose between several backend (gepetto-gui or
    meshcat).

    .. note::
        Replay speed is independent of the platform (windows, linux...) and
        available CPU power.

    :param trajectory_data:
        .. raw:: html

            List of trajectory dictionary with keys:

        - **'evolution_robot':** list of State objects of increasing time.
        - **'robot':** Jiminy robot. None if omitted.
        - **'use_theoretical_model':** whether to use the theoretical or actual
          model.
    :param replay_speed: Speed ratio of the simulation.
                         Optional: 1.0 by default.
    :param record_video_path: Fullpath location where to save generated video
                              (.mp4 extension is: mandatory). Must be specified
                              to enable video recording. None to disable.
                              Optional: None by default.
    :param viewers: Already instantiated viewers, associated one by one in
                    order to each trajectory data. None to disable.
                    Optional: None by default.
    :param start_paused: Start the simulation is pause, waiting for keyboard
                         input before starting to play the trajectories.
                         Optional: False by default.
    :param wait_for_client: Wait for the client to finish loading the meshes
                            before starting.
                            Optional: True by default.
    :param travelling_frame: Name of the frame to automatically follow with the
                             camera. None to disable.
                             Optional: None by default.
    :param camera_xyzrpy: Tuple position [X, Y, Z], rotation [Roll, Pitch, Yaw]
                          corresponding to the absolute pose of the camera
                          during replay, if travelling is disable, or the
                          relative pose wrt the tracked frame otherwise. None
                          to disable.
                          Optional:None by default.
    :param xyz_offset: Constant translation of the root joint in world frame.
                       None to disable.
                       Optional: None by default.
    :param urdf_rgba: RGBA code defining the color of the model. It is the same
                      for each link. None to disable.
                      Optional: Original colors of each link. No alpha.
    :param backend: Backend, one of 'meshcat' or 'gepetto-gui'. If None,
                    'meshcat' is used in notebook environment and 'gepetto-gui'
                    otherwise.
                    Optional: None by default.
    :param window_name: Name of the Gepetto-viewer's window in which to display
                        the robot.
                        Optional: Common default name if omitted.
    :param scene_name: Name of the Gepetto-viewer's scene in which to display
                       the robot.
                       Optional: Common default name if omitted.
    :param close_backend: Close backend automatically at exit.
                          Optional: Enable by default if not (presumably)
                          available beforehand.
    :param delete_robot_on_close: Whether or not to delete the robot from the
                                  viewer when closing it.
                                  Optional: True by default.
    :param verbose: Add information to keep track of the process.
                    Optional: True by default.

    :returns: List of viewers used to play the trajectories.
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
            robot_name = f"robot_{i}"
            use_theoretical_model = trajectory_data[i]['use_theoretical_model']
            viewer = Viewer(
                robot,
                use_theoretical_model=use_theoretical_model,
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
    if wait_for_client and record_video_path is None:
        if Viewer.backend.startswith('meshcat'):
            if verbose and not is_notebook():
                print("Waiting for meshcat client in browser to connect: "
                      f"{Viewer._backend_obj.gui.url()}")
            Viewer.wait(require_client=True)
            if verbose and not is_notebook():
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
        for traj in trajectory_data:
            data_orig = traj['evolution_robot']
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
        def replay_thread(viewer, *args):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            viewer.replay(*args)

        # Play trajectories with multithreading
        threads = []
        for i in range(len(trajectory_data)):
            threads.append(Thread(
                target=replay_thread,
                args=(viewers[i],
                      trajectory_data[i]['evolution_robot'],
                      replay_speed,
                      xyz_offset[i],
                      wait_for_client)))
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


def play_logfiles(robots: Union[List[jiminy.Robot], jiminy.Robot],
                  logs_data: Union[List[Dict[str, np.ndarray]],
                                   Dict[str, np.ndarray]],
                  **kwargs) -> List[Viewer]:
    """Play the content of a logfile in a viewer.

    This method simply formats the data then calls play_trajectories.

    :param robots: Either a single robot, or a list of robot for each log data.
    :param logs_data: Either a single dictionary, or a list of dictionaries of
                      simulation data log.
    :param kwargs: Keyword arguments to forward to `play_trajectories` method.
    """
    # Reformat everything as lists
    if not isinstance(logs_data, list):
        logs_data = [logs_data]
    if not isinstance(robots, list):
        robots = [robots] * len(logs_data)

    # For each pair (log, robot), extract a trajectory object for
    # `play_trajectories`
    trajectories = [extract_viewer_data_from_log(log, robot)
                    for log, robot in zip(logs_data, robots)]

    # Finally, play the trajectories
    return play_trajectories(trajectories, **kwargs)
