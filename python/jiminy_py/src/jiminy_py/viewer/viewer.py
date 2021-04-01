import os
import re
import io
import sys
import time
import shutil
import base64
import atexit
import logging
import pathlib
import tempfile
import subprocess
import webbrowser
import multiprocessing
import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import wraps
from bisect import bisect_right
from threading import Lock
from typing import Optional, Union, Sequence, Tuple, Callable

import psutil
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from typing_extensions import TypedDict

import zmq
import meshcat.transformations as mtf
from panda3d_viewer import (
    Viewer as Panda3dViewer, ViewerConfig as Panda3dViewerConfig)
from panda3d_viewer.viewer_app import ViewerApp as Panda3dApp
from panda3d_viewer.viewer_errors import ViewerClosedError

import pinocchio as pin
from pinocchio import SE3, SE3ToXYZQUAT
from pinocchio.rpy import rpyToMatrix, matrixToRpy
from pinocchio.visualize import GepettoVisualizer

from .. import core as jiminy
from ..state import State
from .meshcat.utilities import interactive_mode
from .meshcat.wrapper import MeshcatWrapper
from .meshcat.meshcat_visualizer import MeshcatVisualizer
from .panda3d.panda3d_visualizer import Panda3dVisualizer


CAMERA_INV_TRANSFORM_PANDA3D = rpyToMatrix(np.array([-np.pi / 2, 0.0, 0.0]))
CAMERA_INV_TRANSFORM_MESHCAT = rpyToMatrix(np.array([-np.pi / 2, 0.0, 0.0]))
DEFAULT_CAMERA_XYZRPY_ABS = [[7.5, 0.0, 1.4], [1.4, 0.0, np.pi / 2]]
DEFAULT_CAMERA_XYZRPY_REL = [[4.5, -4.5, 1.5], [1.3, 0.0, 0.8]]

DEFAULT_CAPTURE_SIZE = 500
DEFAULT_WATERMARK_MAXSIZE = (150, 150)


# Determine set the of available backends
backends_available = {
    'meshcat': MeshcatVisualizer, 'panda3d': Panda3dVisualizer}
if __import__('platform').system() == 'Linux':
    import importlib
    if (importlib.util.find_spec("gepetto") is not None and
            importlib.util.find_spec("omniORB") is not None):
        backends_available['gepetto-gui'] = GepettoVisualizer


def default_backend() -> str:
    """Determine the default backend viewer, depending on the running
    environment and the set of available backends.

    Meshcat will always be prefered in interactive mode, i.e. in Jupyter
    notebooks, while Panda3d otherwise, unless there is some clues that no
    X11-server is available on Linux. In such a case, it fallbacks to Meshcat
    for now, since Nvidia EGL support without X-server of Panda3d is
    implemented but not provided with the official wheels distributed on Pypi
    so far. As a result, Panda3d would work, but relying on software rendering,
    which is know to be unefficient. On the contrary, Meshcat supports Nvidia
    EGL through bundled Chromium web-browser, but only on Linux-based OS.
    """
    if interactive_mode():
        return 'meshcat'
    else:
        if not sys.platform.startswith('linux') or os.environ.get('DISPLAY'):
            return 'panda3d'
        else:
            return 'meshcat'


def _get_backend_exceptions(
        backend: Optional[str] = None) -> Sequence[Exception]:
    """Get the list of exceptions that may be raised by a given backend.
    """
    if backend is None:
        backend = default_backend()
    if backend == 'gepetto-gui':
        import gepetto
        import omniORB
        return (omniORB.CORBA.COMM_FAILURE,
                omniORB.CORBA.TRANSIENT,
                gepetto.corbaserver.gepetto.Error)
    elif backend == 'panda3d':
        return (ViewerClosedError,)
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
    """Function to provide cross-platform time sleep with maximum accuracy.

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
                 proc: Union[
                     multiprocessing.Process, subprocess.Popen,
                     psutil.Process, Panda3dApp],
                 kill_at_exit: bool = False):
        self._proc = proc
        # Make sure the process is killed at Python exit
        if kill_at_exit:
            atexit.register(self.kill)

    def is_parent(self) -> bool:
        return not isinstance(self._proc, psutil.Process)

    def is_alive(self) -> bool:
        if isinstance(self._proc, subprocess.Popen):
            return self._proc.poll() is None
        elif isinstance(self._proc, multiprocessing.Process):
            return self._proc.is_alive()
        elif isinstance(self._proc, Panda3dApp):
            return True  # TODO
        elif isinstance(self._proc, psutil.Process):
            try:
                return self._proc.status() in [
                    psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]
            except psutil.NoSuchProcess:
                return False

    def wait(self, timeout: Optional[float] = None) -> bool:
        if isinstance(self._proc, multiprocessing.Process):
            return self._proc.join(timeout)
        elif isinstance(self._proc, Panda3dApp):
            return None  # TODO
        elif isinstance(self._proc, (
                subprocess.Popen, psutil.Process)):
            return self._proc.wait(timeout)

    def kill(self) -> None:
        if self.is_parent() and self.is_alive():
            if isinstance(self._proc, Panda3dApp):
                pass  # TODO
            else:
                # Try to terminate cleanly
                self._proc.terminate()
                try:
                    self.wait(timeout=0.5)
                except (subprocess.TimeoutExpired,
                        multiprocessing.TimeoutError):
                    pass

                # Force kill if necessary and reap the zombies
                try:
                    psutil.Process(self._proc.pid).kill()
                    os.waitpid(self._proc.pid, 0)
                    os.waitpid(os.getpid(), 0)
                except (psutil.NoSuchProcess, ChildProcessError):
                    pass
                multiprocessing.active_children()


Tuple3FType = Union[Tuple[float, float, float], np.ndarray]
Tuple4FType = Union[Tuple[float, float, float, float], np.ndarray]
CameraPoseType = Tuple[Optional[Tuple3FType], Optional[Tuple3FType]]


class CameraMotionBreakpointType(TypedDict, total=True):
    """Time
    """
    t: float
    """Absolute pose of the camera, as a tuple position [X, Y, Z], rotation
    [Roll, Pitch, Yaw].
    """
    pose: Tuple[Tuple3FType, Tuple3FType]


CameraMotionType = Sequence[CameraMotionBreakpointType]


class Viewer:
    """ TODO: Write documentation.

    .. note::
        The environment variable 'JIMINY_VIEWER_INTERACTIVE_DISABLE' can be
        used to force disabling interactive display.
    """
    backend = default_backend()
    _backend_obj = None
    _backend_exceptions = _get_backend_exceptions()
    _backend_proc = None
    _backend_robot_names = set()
    _backend_robot_colors = {}
    _has_gui = False
    _camera_motion = None
    _camera_travelling = None
    _camera_xyzrpy = deepcopy(DEFAULT_CAMERA_XYZRPY_ABS)
    _lock = Lock()  # Unique lock for every viewer in same thread by default

    def __init__(self,
                 robot: jiminy.Robot,
                 use_theoretical_model: bool = False,
                 urdf_rgba: Optional[Tuple4FType] = None,
                 lock: Optional[Lock] = None,
                 backend: Optional[str] = None,
                 open_gui_if_parent: Optional[bool] = None,
                 delete_robot_on_close: bool = False,
                 robot_name: Optional[str] = None,
                 window_name: str = 'jiminy',
                 scene_name: str = 'world',
                 **kwargs):
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
        :param kwargs: Unused extra keyword arguments to enable forwarding.
        """
        # Handling of default arguments
        if robot_name is None:
            uniq_id = next(tempfile._get_candidate_names())
            robot_name = "_".join(("robot", uniq_id))

        # Backup some user arguments
        self.urdf_rgba = urdf_rgba
        self.robot_name = robot_name
        self.scene_name = scene_name
        self.window_name = window_name
        self.use_theoretical_model = use_theoretical_model
        self._lock = lock if lock is not None else Viewer._lock
        self.delete_robot_on_close = delete_robot_on_close

        # Select the desired backend
        if backend is None:
            backend = Viewer.backend
        else:
            backend = backend.lower()  # Make sure backend's name is lowercase
            if backend not in backends_available:
                if backend.startswith('gepetto'):
                    backend = 'gepetto-gui'
                else:
                    raise ValueError("%s backend not available." % backend)

        # Update the backend currently running, if any
        if Viewer.backend != backend and Viewer.is_alive():
            Viewer.close()
            logging.warning("Different backend already running. Closing it...")
        Viewer.backend = backend

        # Configure exception handling
        Viewer._backend_exceptions = _get_backend_exceptions(backend)

        # Check if the backend is still working, not just alive, if any
        if Viewer.is_alive():
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

        # Reset some class attribute if backend not available
        if not is_backend_running:
            Viewer._has_gui = False
            Viewer._backend_robot_names.clear()
            Viewer._backend_robot_colors.clear()
            Viewer._camera_xyzrpy = deepcopy(DEFAULT_CAMERA_XYZRPY_ABS)
            Viewer.detach_camera()

        # Make sure that the windows, scene and robot names are valid
        if scene_name == window_name:
            raise ValueError(
                "The name of the scene and window must be different.")

        if robot_name in Viewer._backend_robot_names:
            raise ValueError(
                "Robot name already exists but must be unique. Please choose "
                "a different one, or close the associated viewer.")

        # Create a unique temporary directory, specific to this viewer instance
        self._tempdir = tempfile.mkdtemp(
            prefix="_".join((window_name, scene_name, robot_name, "")))

        # Access the current backend or create one if none is available
        self.__is_open = False
        self.is_backend_parent = False
        try:
            # Connect viewer backend
            if not Viewer.is_alive():
                # Handling of default argument(s)
                open_gui = open_gui_if_parent
                if open_gui is None:
                    # Opening a new display cell automatically if there is no
                    # other display cell already opened. The user is probably
                    # expecting a display cell to open in such cases, but there
                    # is no fixed rule.
                    open_gui = interactive_mode() and \
                        not Viewer._backend_obj.comm_manager.n_comm

                # Start viewer backend
                Viewer.__connect_backend(
                    start_if_needed=True, open_gui=open_gui)

                # Update some flags
                self.is_backend_parent = True
            self._gui = Viewer._backend_obj.gui
            self.__is_open = True

            # Keep track of the backend process associated to the viewer.
            # The destructor of this instance must adapt its behavior to the
            # case where the backend process has changed in the meantime.
            self._backend_proc = Viewer._backend_proc

            # Load the robot
            self._setup(robot, self.urdf_rgba)
            Viewer._backend_robot_names.add(self.robot_name)
            Viewer._backend_robot_colors.update({
                self.robot_name: self.urdf_rgba})
        except Exception as e:
            raise RuntimeError(
                "Impossible to create backend or connect to it.") from e

        # Set default camera pose
        if self.is_backend_parent:
            self.set_camera_transform()

        # Refresh the viewer since the positions of the meshes and their
        # visibility mode are not properly set at this point.
        self.refresh(force_update_visual=True, force_update_collision=True)

    def __del__(self) -> None:
        """Destructor.

        .. note::
            It automatically close the viewer before being garbage collected.
        """
        self.close()

    def __must_be_open(fct: Callable) -> Callable:
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

    @__must_be_open
    def _setup(self,
               robot: jiminy.Robot,
               urdf_rgba: Optional[Tuple4FType] = None) -> None:
        """Load (or reload) robot in viewer.

        .. note::
            This method must be called after calling `engine.reset` since at
            this point the viewer has dangling references to the collision
            model and data of robot. Indeed, a new robot is generated at each
            reset to add some stockasticity to the mass distribution and some
            other parameters. This is done automatically if  one is using
            `simulator.Simulator` instead of `jiminy_py.core.Engine` directly.

        :param robot: Jiminy.Robot to display.
        :param urdf_rgba: RGBA color to use to display this robot, as a list
                          of 4 floating-point values between 0.0 and 1.0.
                          Optional: It will override the original color of the
                          meshes if specified.
        """
        # Backup desired color
        self.urdf_rgba = urdf_rgba

        # Generate colorized URDF file if using gepetto-gui backend, since it
        # is not supported by default, because of memory optimizations.
        self.urdf_path = os.path.realpath(robot.urdf_path)
        if Viewer.backend == 'gepetto-gui':
            if self.urdf_rgba is not None:
                assert len(self.urdf_rgba) == 4
                alpha = self.urdf_rgba[3]
                self.urdf_path = Viewer._get_colorized_urdf(
                    robot, self.urdf_rgba[:3], self._tempdir)
            else:
                alpha = 1.0

        # Extract the right Pinocchio model
        if self.use_theoretical_model:
            pinocchio_model = robot.pinocchio_model_th
            pinocchio_data = robot.pinocchio_data_th
        else:
            pinocchio_model = robot.pinocchio_model
            pinocchio_data = robot.pinocchio_data

        # Create robot visual model
        visual_model = pin.buildGeomFromUrdf(
            pinocchio_model, self.urdf_path, pin.GeometryType.VISUAL,
            robot.mesh_package_dirs)

        # Create backend wrapper to get (almost) backend-independent API
        self._client = backends_available[Viewer.backend](
            pinocchio_model, robot.collision_model, visual_model)
        self._client.data = pinocchio_data
        self._client.collision_data = robot.collision_data

        # Delete existing robot, if any
        robot_node_path = '/'.join((self.scene_name, self.robot_name))
        Viewer._delete_nodes_viewer([robot_node_path])

        # Create the scene and load robot
        if Viewer.backend == 'gepetto-gui':
            # Initialize the viewer
            self._client.initViewer(
                viewer=Viewer._backend_obj, windowName=self.window_name,
                sceneName=self.scene_name, loadModel=False)

            # Add missing scene elements
            self._gui.addFloor('/'.join((self.scene_name, "floor")))
            self._gui.addLandmark(self.scene_name, 0.1)

            # Load the robot
            self._client.loadViewerModel(rootNodeName=self.robot_name)

            # Set robot transparency
            try:
                self._gui.setFloatProperty(robot_node_path, 'Alpha', alpha)
            except Viewer._backend_exceptions:
                # Old Gepetto versions do no have 'Alpha' attribute but
                # 'Transparency'.
                self._gui.setFloatProperty(
                    robot_node_path, 'Transparency', 1 - alpha)
        else:
            # Initialize the viewer
            self._client.initViewer(viewer=self._gui, loadModel=False)

            # Load the robot
            robot_node_path = '/'.join((self.scene_name, self.robot_name))
            self._client.loadViewerModel(
                rootNodeName=robot_node_path, color=urdf_rgba)

    @staticmethod
    def open_gui(start_if_needed: bool = False) -> bool:
        """Open a new viewer graphical interface.

        .. note::
            This method does nothing when using Gepetto-gui backend because
            its lifetime is tied to the graphical interface.

        .. note::
            Only one graphical interface can be opened locally for efficiency.
        """
        # Start backend if needed
        if not Viewer.is_alive():
            Viewer.__connect_backend(start_if_needed)

        # If a graphical window is already open, do nothing
        if Viewer._has_gui:
            return

        if Viewer.backend == 'gepetto-gui':
            # No instance is considered manager of the unique window
            pass
        elif Viewer.backend == 'panda3d':
            Viewer._backend_obj._app.open_window()
        elif Viewer.backend == 'meshcat':
            viewer_url = Viewer._backend_obj.gui.url()

            if interactive_mode():
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

                if interactive_mode() == 1:
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
                        "No browser available for display. Please install one "
                        "manually.")
                    return  # Skip waiting since there is nothing to wait for

        # Wait to finish loading
        Viewer.wait(require_client=True)

        # There is at least one graphical window at this point
        Viewer._has_gui = True

    @staticmethod
    @__must_be_open
    def wait(require_client: bool = False) -> None:
        """Wait for all the meshes to finish loading in every clients.

        :param require_client: Wait for at least one client to be available
                               before checking for mesh loading.
        """
        if Viewer.backend == 'meshcat':
            # Only Meshcat is asynchronous. Note that Gepetto-gui can be
            # updated asynchronously, but it is more difficult to manage for
            # no real advantage.
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
            if self is None:
                self = Viewer

            if self is Viewer:
                # NEVER closing backend automatically if closing instances,
                # even for the parent. It will be closed at Python exit
                # automatically. One must call `Viewer.close` to do otherwise.
                Viewer._backend_robot_names.clear()
                Viewer._backend_robot_colors.clear()
                Viewer.detach_camera()
                if Viewer.is_alive():
                    if Viewer.backend == 'meshcat':
                        Viewer._backend_obj.close()
                        recorder_proc = Viewer._backend_obj.recorder.proc
                        _ProcessWrapper(recorder_proc).kill()
                    else:
                        Viewer._backend_obj._app.destroy()
                    Viewer._backend_proc.kill()
                Viewer._backend_obj = None
                Viewer._backend_proc = None
                Viewer._has_gui = False
            else:
                # Disable travelling if associated with this viewer instance
                if (Viewer._camera_travelling is not None and
                        Viewer._camera_travelling['viewer'] is self):
                    Viewer.detach_camera()

                # Check if the backend process has changed, which may happend
                # if it has been closed manually in the meantime. If so, there
                # is nothing left to do.
                if Viewer._backend_proc is not self._backend_proc:
                    return

                # Make sure zmq does not hang
                if Viewer.backend == 'meshcat' and Viewer.is_alive():
                    Viewer._backend_obj.gui.window.zmq_socket.RCVTIMEO = 50

                # Consider that the robot name is now available, no matter
                # whether the robot has actually been deleted or not.
                Viewer._backend_robot_names.discard(self.robot_name)
                Viewer._backend_robot_colors.pop(self.robot_name)
                if self.delete_robot_on_close:
                    Viewer._delete_nodes_viewer(
                        ['/'.join((self.scene_name, self.robot_name))])

                if Viewer.backend == 'meshcat':
                    Viewer._backend_obj.gui.window.zmq_socket.RCVTIMEO = -1

                if self._tempdir.startswith(tempfile.gettempdir()):
                    try:
                        shutil.rmtree(self._tempdir)
                    except FileNotFoundError:
                        pass
        except Exception:  # Do not fail under any circumstances
            pass

        # At this point, consider the viewer has been closed, no matter what
        self.__is_open = False

    @staticmethod
    def _get_colorized_urdf(robot: jiminy.Robot,
                            rgb: Tuple3FType,
                            output_root_path: Optional[str] = None) -> str:
        """Generate a unique colorized URDF for a given robot model.

        .. note::
            Multiple identical URDF model of different colors can be loaded in
            Gepetto-viewer this way.

        :param robot: Jiminy.Robot already initialized for the desired URDF.
        :param rgb: RGB code defining the color of the model. It is the same
                    for each link.
        :param output_root_path: Root directory of the colorized URDF data.
                                 Optional: temporary directory by default.

        :returns: Full path of the colorized URDF file.
        """
        # Get the URDF path and mesh directory search paths if any
        urdf_path = robot.urdf_path
        mesh_package_dirs = robot.mesh_package_dirs

        # Define color tag and string representation
        color_tag = " ".join(map(str, list(rgb) + [1.0]))
        color_str = "_".join(map(str, list(rgb) + [1.0]))

        # Create the output directory
        if output_root_path is None:
            output_root_path = tempfile.mkdtemp()
        colorized_data_dir = os.path.join(
            output_root_path, f"colorized_urdf_rgb_{color_str}")
        os.makedirs(colorized_data_dir, exist_ok=True)
        colorized_urdf_path = os.path.join(
            colorized_data_dir, os.path.basename(urdf_path))

        # Parse the URDF file
        tree = ET.parse(robot.urdf_path)
        root = tree.getroot()

        # Update mesh fullpath and material color for every visual
        for visual in root.iterfind('./link/visual'):
            # Get mesh full path
            for geom in visual.iterfind('geometry'):
                # Get mesh path if any, otherwise skip the geometry
                mesh_descr = geom.find('mesh')
                if mesh_descr is None:
                    continue
                mesh_fullpath = mesh_descr.get('filename')

                # Make sure mesh path is fully qualified and exists
                mesh_realpath = None
                if mesh_fullpath.startswith('package://'):
                    for mesh_dir in mesh_package_dirs:
                        mesh_searchpath = os.path.join(
                            mesh_dir, mesh_fullpath[10:])
                        if os.path.exists(mesh_searchpath):
                            mesh_realpath = mesh_searchpath
                            break
                else:
                    mesh_realpath = mesh_fullpath
                assert mesh_realpath is not None, (
                    f"Invalid mesh path '{mesh_fullpath}'.")

                # Copy original meshes to temporary directory
                colorized_mesh_fullpath = os.path.join(
                    colorized_data_dir, mesh_realpath[1:])
                colorized_mesh_path = os.path.dirname(colorized_mesh_fullpath)
                if not os.access(colorized_mesh_path, os.F_OK):
                    os.makedirs(colorized_mesh_path)
                shutil.copy2(mesh_realpath, colorized_mesh_fullpath)

                # Update mesh fullpath
                geom.find('mesh').set('filename', mesh_realpath)

            # Update color tag if any, create one otherwise
            material = visual.find('material')
            if material is not None:
                name = material.get('name')
                if name is not None:
                    material = root.find(f"./material[@name='{name}']")
                material.find('color').set('rgba', color_tag)
            else:
                material = ET.SubElement(visual, 'material', name='')
                ET.SubElement(material, 'color', rgba=color_tag)

        # Write on disk the generated URDF file
        tree = ET.ElementTree(root)
        tree.write(colorized_urdf_path)

        return colorized_urdf_path

    @staticmethod
    def __connect_backend(start_if_needed: bool = False,
                          open_gui: Optional[bool] = None,
                          close_at_exit: bool = True,
                          timeout: int = 2000) -> None:
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
        if Viewer.backend == 'gepetto-gui':
            from gepetto.corbaserver.client import Client as gepetto_client

            if open_gui is not None and not open_gui:
                logger.warning(
                    "This option is not available for Gepetto-gui.")
            open_gui = False

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
                client, proc = _gepetto_client_connect(get_proc_info=True)
            except Viewer._backend_exceptions:
                try:
                    client, proc = _gepetto_client_connect(get_proc_info=True)
                except Viewer._backend_exceptions:
                    if start_if_needed:
                        FNULL = open(os.devnull, 'w')
                        proc = subprocess.Popen(
                            ['gepetto-gui'], shell=False, stdout=FNULL,
                            stderr=FNULL)
                        proc = _ProcessWrapper(proc, close_at_exit)
                        # Must try at least twice for robustness
                        is_connected = False
                        for _ in range(max(2, int(timeout / 200))):
                            time.sleep(0.2)
                            try:
                                client = _gepetto_client_connect()
                                is_connected = True
                                continue
                            except Viewer._backend_exceptions:
                                pass
                        if not is_connected:
                            raise RuntimeError(
                                "Impossible to open Gepetto-viewer.")
                    else:
                        raise RuntimeError(
                            "No backend server to connect to but "
                            "'start_if_needed' is set to False")
        elif Viewer.backend == 'panda3d':
            # handle default argument(s)
            if open_gui is None:
                open_gui = True

            # Make sure that creating a new client is allowed
            if not start_if_needed:
                raise RuntimeError(
                    "'panda3d' backend does not support connecting to already "
                    "running client.")

            # Instantiate new client with onscreen rendering enabled.
            # Note that it fallbacks to software rendering if necessary.
            config = Panda3dViewerConfig()
            config.set_window_size(DEFAULT_CAPTURE_SIZE, DEFAULT_CAPTURE_SIZE)
            config.set_window_fixed(False)
            config.enable_antialiasing(True, multisamples=4)
            config.enable_shadow(True)
            config.enable_lights(True)
            config.enable_hdr(False)
            config.enable_fog(False)
            config.show_axes(True)
            config.show_grid(False)
            config.show_floor(True)
            config.set_value('framebuffer-software', '0')
            config.set_value('framebuffer-hardware', '0')
            config.set_value('load-display', 'pandagl')
            config.set_value('aux-display',
                             'p3headlessgl'
                             '\naux-display pandadx9'
                             '\naux-display pandadx8'
                             '\naux-display p3tinydisplay')
            client = Panda3dViewer(
                window_type='onscreen', window_title='jiminy', config=config)
            client.gui = client  # The gui is the client itself for now

            proc = _ProcessWrapper(client._app, close_at_exit)
        else:
            # handle default argument(s)
            if open_gui is None:
                open_gui = True

            # List of connections likely to correspond to Meshcat servers
            meshcat_candidate_conn = []
            for conn in psutil.net_connections("tcp4"):
                if conn.status == 'LISTEN' and conn.laddr.ip == '127.0.0.1':
                    try:
                        cmdline = psutil.Process(conn.pid).cmdline()
                        if not cmdline:
                            continue
                        if 'python' in cmdline[0] or 'meshcat' in cmdline[-1]:
                            meshcat_candidate_conn.append(conn)
                    except psutil.AccessDenied:
                        pass

            # Exclude ipython kernel ports from the look up because sending a
            # message on ipython ports will throw a low-level exception, that
            # is not blocking on Jupyter, but is on Google Colab.
            excluded_ports = []
            if interactive_mode():
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

            # Check if backend server was found
            if not start_if_needed and zmq_url is None:
                raise RuntimeError(
                    "No backend server to connect to but 'start_if_needed' is "
                    "set to False")

            # Create a meshcat server if needed and connect to it
            client = MeshcatWrapper(zmq_url)
            if client.server_proc is None:
                proc = psutil.Process(conn.pid)
            else:
                proc = client.server_proc
            proc = _ProcessWrapper(proc, close_at_exit)

        # Update global state
        Viewer._backend_obj = client
        Viewer._backend_proc = proc

        # Open gui if requested
        if open_gui:
            Viewer.open_gui()

    @staticmethod
    @__must_be_open
    def _delete_nodes_viewer(nodes_path: Sequence[str]) -> None:
        """Delete a 'node' in Gepetto-viewer.

        .. note::
            Be careful, one must specify the full path of a node, including all
            parent group, but without the window name, ie
            'scene_name/robot_name' to delete the robot.

        :param nodes_path: Full path of the node to delete
        """
        if Viewer.backend == 'gepetto-gui':
            for node_path in nodes_path:
                Viewer._backend_obj.gui.deleteNode(node_path, True)
        elif Viewer.backend == 'panda3d':
            for node_path in nodes_path:
                try:
                    Viewer._backend_obj.gui.remove_group(node_path)
                except KeyError:
                    pass
        else:
            for node_path in nodes_path:
                Viewer._backend_obj.gui[node_path].delete()

    @staticmethod
    @__must_be_open
    def set_watermark(img_fullpath: Optional[str] = None,
                      width: Optional[int] = None,
                      height: Optional[int] = None) -> None:
        """Insert desired watermark on bottom left corner of the window.

        .. note::
            The relative width and height cannot exceed 20% of the visible
            area, othewise it will be rescaled.

        .. note::
            Gepetto-gui is not supported by this method and will never be.

        :param img_fullpath: Full path of the image to use as watermark.
                             Meshcat supports format '.png', '.jpeg' or 'svg',
                             while Panda3d only supports '.png' and '.jpeg' for
                             now. None or empty string to disable.
                             Optional: None by default.
        :param width: Desired width for the image. None to not rescale the
                      image manually.
                      Optional: None by default.
        :param height: Desired height for the image. None to not rescale the
                       image manually.
                       Optional: None by default.
        """
        if Viewer.backend == 'gepetto-gui':
            logger.warning(
                "Adding watermark is not available for Gepetto-gui.")
        elif Viewer.backend == 'panda3d':
            Viewer._backend_obj._app.set_watermark(img_fullpath, width, height)
        else:
            width = width or DEFAULT_WATERMARK_MAXSIZE[0]
            height = height or DEFAULT_WATERMARK_MAXSIZE[1]
            if img_fullpath is None or img_fullpath == "":
                Viewer._backend_obj.remove_watermark()

    @staticmethod
    @__must_be_open
    def set_legend(labels: Optional[Sequence[str]] = None) -> None:
        """Insert legend on top left corner of the window.

        .. note::
            Make sure to have specified different colors for each robot on the
            scene, since it will be used as marker on the legend.

        .. note::
            Gepetto-gui is not supported by this method and will never be.

        :param labels: Sequence of strings whose length must be consistent
                       with the number of robots on the scene. None to disable.
                       Optional: None by default.
        """
        # Make sure number of labels is consistent with number of robots
        if labels is not None:
            assert len(labels) == len(Viewer._backend_robot_colors)

        if Viewer.backend == 'gepetto-gui':
            logger.warning("Adding legend is not available for Gepetto-gui.")
        elif Viewer.backend == 'panda3d':
            if labels is None:
                items = None
            else:
                items = dict(zip(
                    labels, Viewer._backend_robot_colors.values()))
            Viewer._backend_obj._app.set_legend(items)
        else:
            if labels is None:
                for robot_name in Viewer._backend_robot_colors.keys():
                    Viewer._backend_obj.remove_legend_item(robot_name)
            else:
                for text, (robot_name, color) in zip(
                        labels, Viewer._backend_robot_colors.items()):
                    rgba = [*[int(e * 255) for e in color[:3]], color[3]]
                    color = f"rgba({','.join(map(str, rgba))}"
                    Viewer._backend_obj.set_legend_item(
                        robot_name, color, text)

    @staticmethod
    @__must_be_open
    def set_clock(time: Optional[float] = None) -> None:
        """Insert clock on bottom right corner of the window.

        .. note::
            Only Panda3d rendering backend is supported by this method.

        :param time: Current time is seconds. None to disable.
                     Optional: None by default.
        """
        if Viewer.backend == 'panda3d':
            Viewer._backend_obj._app.set_clock(time)
        else:
            logger.warning("Adding clock is only available for Panda3d.")

    @__must_be_open
    def set_camera_transform(self,
                             position: Optional[Tuple3FType] = None,
                             rotation: Optional[Tuple3FType] = None,
                             relative: Optional[str] = None) -> None:
        """Apply transform to the camera pose.

        :param position: Position [X, Y, Z] as a list or 1D array. None to not
                         update it.
                         Optional: None by default.
        :param rotation: Rotation [Roll, Pitch, Yaw] as a list or 1D np.array.
                         None to note update it.
                         Optional: None by default.
        :param relative:
            .. raw:: html

                How to apply the transform:

            - **None:** absolute
            - **'camera':** relative to the current camera pose
            - **other string:** relative to a robot frame, not accounting for
              the rotation (travelling)
        """
        # Handling of position and rotation arguments
        if position is None:
            position = Viewer._camera_xyzrpy[0]
        if rotation is None:
            rotation = Viewer._camera_xyzrpy[1]
        position, rotation = np.asarray(position), np.asarray(rotation)

        # Compute associated rotation matrix
        rotation_mat = rpyToMatrix(rotation)

        # Compute the relative transformation if applicable
        if relative == 'camera':
            H_orig = SE3(rpyToMatrix(np.asarray(Viewer._camera_xyzrpy[1])),
                         np.asarray(Viewer._camera_xyzrpy[0]))
        elif relative is not None:
            # Get the body position, not taking into account the rotation
            body_id = self._client.model.getFrameId(relative)
            try:
                body_transform = self._client.data.oMf[body_id]
            except IndexError:
                raise ValueError("'relative' set to non-existing frame.")
            H_orig = SE3(np.eye(3), body_transform.translation)

        # Compute the absolute transformation
        if relative is not None:
            H_abs = SE3(rotation_mat, position)
            H_abs = H_orig * H_abs
            position = H_abs.translation
            return self.set_camera_transform(position, rotation)

        # Perform the desired transformation
        if Viewer.backend == 'gepetto-gui':
            H_abs = SE3(rotation_mat, position)
            self._gui.setCameraTransform(
                self._client.windowID, SE3ToXYZQUAT(H_abs).tolist())
        elif Viewer.backend == 'panda3d':
            rotation_panda3d = pin.Quaternion(
                rotation_mat @ CAMERA_INV_TRANSFORM_PANDA3D).coeffs()
            self._gui._app.set_camera_transform(position, rotation_panda3d)
        elif Viewer.backend == 'meshcat':
            # Meshcat camera is rotated by -pi/2 along Roll axis wrt the
            # usual convention in robotics.
            position_meshcat = CAMERA_INV_TRANSFORM_MESHCAT @ position
            rotation_meshcat = matrixToRpy(
                CAMERA_INV_TRANSFORM_MESHCAT @ rotation_mat)
            self._gui["/Cameras/default/rotated/<object>"].\
                set_transform(mtf.compose_matrix(
                    translate=position_meshcat, angles=rotation_meshcat))

        # Backup updated camera pose
        Viewer._camera_xyzrpy[0] = position.copy()
        Viewer._camera_xyzrpy[1] = rotation.copy()

    @staticmethod
    def register_camera_motion(camera_motion: CameraMotionType) -> None:
        """Register camera motion. It will be used later by `replay` to set the
        absolute or relative camera pose, depending on whether or not
        travelling is enable.

        :param camera_motion:
            Camera breakpoint poses over time, as a list of
            `CameraMotionBreakpointType` dict. Here is an example::

                [{'t': 0.00,
                  'pose': ([3.5, 0.0, 1.4], [1.4, 0.0, np.pi / 2])},
                 {'t': 0.33,
                  'pose': ([4.5, 0.0, 1.4], [1.4, np.pi / 6, np.pi / 2])},
                 {'t': 0.66,
                  'pose': ([8.5, 0.0, 1.4], [1.4, np.pi / 3, np.pi / 2])},
                 {'t': 1.00,
                  'pose': ([9.5, 0.0, 1.4], [1.4, np.pi / 2, np.pi / 2])}]
        """
        t_camera = np.asarray([
            camera_break['t'] for camera_break in camera_motion])
        interp_kind = 'linear' if len(t_camera) < 4 else 'cubic'
        camera_xyz = np.stack([camera_break['pose'][0]
                               for camera_break in camera_motion], axis=0)
        camera_motion_xyz = interp1d(
            t_camera, camera_xyz,
            kind=interp_kind, bounds_error=False,
            fill_value=(camera_xyz[0], camera_xyz[-1]), axis=0)
        camera_rpy = np.stack([camera_break['pose'][1]
                               for camera_break in camera_motion], axis=0)
        camera_motion_rpy = interp1d(
            t_camera, camera_rpy,
            kind=interp_kind, bounds_error=False,
            fill_value=(camera_rpy[0], camera_rpy[-1]), axis=0)
        Viewer._camera_motion = lambda t: [
            camera_motion_xyz(t), camera_motion_rpy(t)]

    @staticmethod
    def remove_camera_motion() -> None:
        """Remove camera motion.
        """
        Viewer._camera_motion = None

    def attach_camera(self,
                      frame: str,
                      camera_xyzrpy: Optional[CameraPoseType] = None) -> None:
        """Attach the camera to a given robot frame.

        Only the position of the frame is taken into account. A custom relative
        pose of the camera wrt to the frame can be further specified.

        :param frame: Frame of the robot to follow with the camera.
        :param camera_xyzrpy: Tuple position [X, Y, Z], rotation
                              [Roll, Pitch, Yaw] corresponding to the relative
                              pose of the camera wrt the tracked frame. None
                              to use default pose.
                              Optional: None by default.
        """
        # Make sure one is not trying to track the camera itself...
        assert frame != 'camera', "Impossible to track the camera itself !"

        # Handling of default camera pose
        if camera_xyzrpy is None:
            camera_xyzrpy = DEFAULT_CAMERA_XYZRPY_REL

        Viewer._camera_travelling = {
            'viewer': self, 'frame': frame, 'pose': camera_xyzrpy}

    @staticmethod
    def detach_camera() -> None:
        """Detach the camera.

        Must be called to undo `attach_camera`, so that it will stop
        automatically tracking a frame.
        """
        Viewer._camera_travelling = None

    @__must_be_open
    def capture_frame(self,
                      width: int = None,
                      height: int = None,
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
        # Check user arguments
        if Viewer.backend == 'gepetto-gui' and (
                width is not None or height is not None):
            logger.warning(
                "Specifying window size is not available for Gepetto-gui.")

            if raw_data:
                raise ValueError(
                    "Raw data mode is only available for Meshcat.")

        if Viewer.backend == 'gepetto-gui':
            # It is not possible to capture frame directly using gepetto-gui,
            # and it is not able to save the frame if the file does not have
            # ".png" extension.
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                self.save_frame(f.name)
                img_obj = Image.open(f.name)
                rgba_array = np.array(img_obj)
        elif Viewer.backend == 'panda3d':
            _width, _height = self._gui._app.getSize()
            if width is None:
                width = _width
            if height is None:
                height = _height
            if _width != width or _height != height:
                self._gui._app.set_window_size(width, height)
            # Call low-level `get_screenshot` directly to get raw buffer
            self._gui._app.step()  # Render the current scene
            buffer = self._gui._app.get_screenshot(
                requested_format='RGB', raw=True)
            array = np.frombuffer(buffer, np.uint8).reshape((height, width, 3))
            return np.flipud(array)
        else:
            # Send capture frame request to the background recorder process
            img_html = Viewer._backend_obj.capture_frame(width, height)

            # Parse the output to remove the html header, and convert it into
            # the desired output format.
            img_data = str.encode(img_html.split(",", 1)[-1])
            img_raw = base64.decodebytes(img_data)
            if raw_data:
                return img_raw
            else:
                img_obj = Image.open(io.BytesIO(img_raw))
                rgba_array = np.array(img_obj)
            return rgba_array[:, :, :-1]

    @__must_be_open
    def save_frame(self,
                   image_path: str,
                   width: int = None,
                   height: int = None) -> None:
        """Save a snapshot in png format.

        :param image_path: Fullpath of the image (.png extension is mandatory)
        :param width: Width for the image in pixels (not available with
                      Gepetto-gui for now). None to keep unchanged.
                      Optional: DEFAULT_CAPTURE_SIZE by default.
        :param height: Height for the image in pixels (not available with
                       Gepetto-gui for now). None to keep unchanged.
                       Optional: DEFAULT_CAPTURE_SIZE by default.
        """
        image_path = str(pathlib.Path(image_path).with_suffix('.png'))
        if Viewer.backend == 'gepetto-gui':
            self._gui.captureFrame(self._client.windowID, image_path)
        elif Viewer.backend == 'panda3d':
            _width, _height = self._gui._app.getSize()
            if width is None:
                width = _width
            if height is None:
                height = _height
            if _width != width or _height != height:
                self._gui._app.set_window_size(width, height)
            self._gui.save_screenshot(image_path)
        else:
            img_data = self.capture_frame(width, height, raw_data=True)
            with open(image_path, "wb") as f:
                f.write(img_data)

    @__must_be_open
    def display_visuals(self, visibility: bool) -> None:
        """Set the visibility of the visual model of the robot.

        :param visibility: Whether to enable or disable display of the visual
                           model.
        """
        self._client.displayVisuals(visibility)
        self.refresh()

    @__must_be_open
    def display_collisions(self, visibility: bool) -> None:
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
        travelling.

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

            if Viewer.backend == 'gepetto-gui':
                for model, data, model_type in zip(
                        model_list, data_list, model_type_list):
                    self._gui.applyConfigurations(
                        [self._client.getViewerNodeName(geom, model_type)
                         for geom in model.geometryObjects],
                        [tuple(SE3ToXYZQUAT(data.oMg[i]))
                         for i, geom in enumerate(model.geometryObjects)])
            elif Viewer.backend == 'panda3d':
                for model, data, model_type in zip(
                        model_list, data_list, model_type_list):
                    name_pose_dict = {}
                    for i, geom in enumerate(model.geometryObjects):
                        oMg = data.oMg[i]
                        x, y, z, qx, qy, qz, qw = SE3ToXYZQUAT(oMg)
                        group, nodeName = self._client.getViewerNodeName(
                            geom, model_type)
                        name_pose_dict[nodeName] = (x, y, z), (qw, qx, qy, qz)
                    self._client.viewer.move_nodes(group, name_pose_dict)
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

            # Update the camera placement if necessary
            if Viewer._camera_travelling is not None:
                if Viewer._camera_travelling['viewer'] is self:
                    self.set_camera_transform(
                        *Viewer._camera_travelling['pose'],
                        relative=Viewer._camera_travelling['frame'])
            elif Viewer._camera_motion is not None:
                self.set_camera_transform()

            # Refreshing viewer backend manually is necessary for gepetto-gui
            if Viewer.backend == 'gepetto-gui':
                self._gui.refresh()

            # Wait for the backend viewer to finish rendering if requested
            if wait:
                Viewer.wait(require_client=False)

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
        pin.framesForwardKinematics(self._client.model, self._client.data, q)
        pin.updateGeometryPlacements(
            self._client.model, self._client.data,
            self._client.collision_model, self._client.collision_data)

        # Refresh the viewer
        self.refresh(wait)

    @__must_be_open
    def replay(self,
               evolution_robot: Sequence[State],
               time_interval: Optional[Union[
                   np.ndarray, Tuple[float, float]]] = (0.0, np.inf),
               speed_ratio: float = 1.0,
               xyz_offset: Optional[np.ndarray] = None,
               enable_clock: bool = False,
               wait: bool = False) -> None:
        """Replay a complete robot trajectory at a given real-time ratio.

        .. warning::
            It will alter original robot data if viewer attribute
            `use_theoretical_model` is false.

        :param evolution_robot: List of State object of increasing time.
        :param time_interval: Specific time interval to replay.
                              Optional: Complete evolution by default [0, inf].
        :param speed_ratio: Real-time factor.
                            Optional: No time dilation by default (1.0).
        :param xyz_offset: Freeflyer position offset. Note that it does not
                           check for the robot actually have a freeflyer.
                           OPtional: None by default.
        :param wait: Whether or not to wait for rendering to finish.
        """
        t = [s.t for s in evolution_robot]
        t_simu = time_interval[0]
        i = bisect_right(t, t_simu)
        init_time = time.time()
        while i < len(evolution_robot):
            try:
                if enable_clock:
                    Viewer.set_clock(t_simu)
                s = evolution_robot[i]
                s_next = evolution_robot[min(i, len(evolution_robot)) - 1]
                ratio = (t_simu - s.t) / (s_next.t - s.t)
                q = pin.interpolate(self._client.model, s.q, s_next.q, ratio)
                if Viewer._camera_motion is not None:
                    Viewer._camera_xyzrpy = Viewer._camera_motion(t_simu)
                self.display(q, xyz_offset, wait)
                t_simu = time_interval[0] + speed_ratio * (
                    time.time() - init_time)
                i = bisect_right(t, t_simu)
                wait = False  # Waiting for the first timestep is enough
                if t_simu > time_interval[1]:
                    break
            except Viewer._backend_exceptions:
                # Make sure the viewer is properly closed if exception is
                # raised during replay.
                Viewer.close()
                return

        # Disable clock after replay if enable and alive
        if Viewer.is_alive():
            Viewer.set_clock()
