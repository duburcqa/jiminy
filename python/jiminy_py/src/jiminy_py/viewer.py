import os
import re
import io
import time
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
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from functools import wraps
from bisect import bisect_right
from threading import Thread, Lock
from itertools import cycle, islice
from scipy.interpolate import interp1d
from typing_extensions import TypedDict
from typing import Optional, Union, Sequence, Tuple, Dict, Callable

import zmq
import meshcat.transformations as mtf

import pinocchio as pin
from pinocchio import SE3, SE3ToXYZQUAT
from pinocchio.rpy import rpyToMatrix, matrixToRpy
from pinocchio.visualize import GepettoVisualizer

from . import core as jiminy
from .state import State
from .meshcat.utilities import interactive_mode
from .meshcat.wrapper import MeshcatWrapper
from .meshcat.meshcat_visualizer import MeshcatVisualizer


CAMERA_INV_TRANSFORM_MESHCAT = rpyToMatrix(np.array([-np.pi / 2, 0.0, 0.0]))
DEFAULT_CAMERA_ABS_XYZRPY = [[7.5, 0.0, 1.4], [1.4, 0.0, np.pi / 2]]
DEFAULT_CAMERA_REL_XYZRPY = [[3.0, -3.0, 1.0], [1.3, 0.0, 0.8]]

DEFAULT_CAPTURE_SIZE = 500
VIDEO_FRAMERATE = 30
VIDEO_SIZE = (1000, 1000)
DEFAULT_WATERMARK_MAXSIZE = (150, 150)

DEFAULT_URDF_COLORS = {
    'green': (0.4, 0.7, 0.3, 1.0),
    'purple': (0.6, 0.0, 0.9, 1.0),
    'orange': (1.0, 0.45, 0.0, 1.0),
    'cyan': (0.2, 0.7, 1.0, 1.0),
    'red': (0.9, 0.15, 0.15, 1.0),
    'yellow': (1.0, 0.7, 0.0, 1.0),
    'blue': (0.25, 0.25, 1.0, 1.0)
}


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
    if interactive_mode() or 'gepetto-gui' not in backends_available:
        return 'meshcat'
    else:
        return 'gepetto-gui'


def _get_backend_exceptions(
        backend: Optional[str] = None) -> Sequence[Exception]:
    """Get the list of exceptions that may be raised by a given backend.
    """
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
            try:
                self.wait(timeout=0.5)
            except (subprocess.TimeoutExpired, multiprocessing.TimeoutError):
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
    backend = _default_backend()
    _backend_obj = None
    _backend_exceptions = _get_backend_exceptions()
    _backend_proc = None
    _backend_robot_names = set()
    _camera_motion = None
    _camera_travelling = None
    _camera_xyzrpy = deepcopy(DEFAULT_CAMERA_ABS_XYZRPY)
    _lock = Lock()  # Unique lock for every viewer in same thread by default

    def __init__(self,
                 robot: jiminy.Robot,
                 use_theoretical_model: bool = False,
                 urdf_rgba: Optional[Tuple4FType] = None,
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

        # Reset some class attribute if backend not available
        if not is_backend_running:
            Viewer._backend_robot_names = set()
            Viewer._camera_xyzrpy = deepcopy(DEFAULT_CAMERA_ABS_XYZRPY)
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
            # Create viewer backend if necessary
            if not Viewer.is_alive():
                Viewer._backend_obj, Viewer._backend_proc = \
                    Viewer.__get_client(start_if_needed=True)
                self.is_backend_parent = Viewer._backend_proc.is_parent()
            self._gui = Viewer._backend_obj.gui
            self.__is_open = True

            # Keep track of the backend process associated to the viewer.
            # The destructor of this instance must adapt its behavior to the
            # case where the backend process has changed in the meantime.
            self._backend_proc = Viewer._backend_proc

            # Load the robot
            self._setup(robot, self.urdf_rgba)
            Viewer._backend_robot_names.add(self.robot_name)

            # Open a gui window in browser, since the server is headless.
            # Note that the scene is created automatically as client level, it
            # is not managed by the server.
            if Viewer.backend.startswith('meshcat'):
                if not interactive_mode():
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
                "Impossible to create backend or connect to it.") from e

        # Set default camera pose
        if self.is_backend_parent:
            self.set_camera_transform()

        # Refresh the viewer since the positions of the meshes and their
        # visibility mode are not properly set at this point.
        self.refresh()

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
        if Viewer.backend.startswith('gepetto'):
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

        # Create the scene and load robot
        if Viewer.backend.startswith('gepetto'):
            # Initialize the viewer
            self._client.initViewer(
                viewer=Viewer._backend_obj, windowName=self.window_name,
                sceneName=self.scene_name, loadModel=False)

            # Add missing scene elements
            if Viewer.backend.startswith('gepetto'):
                self._gui.addFloor('/'.join((self.scene_name, "floor")))
                self._gui.addLandmark(self.scene_name, 0.1)

            # Delete existing robot, if any
            Viewer._delete_nodes_viewer(
                ['/'.join((self.scene_name, self.robot_name))])

            # Load the robot
            self._client.loadViewerModel(rootNodeName=self.robot_name)
            robot_node_path = '/'.join((self.scene_name, self.robot_name))
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
            if not Viewer.is_alive():
                Viewer._backend_obj, Viewer._backend_proc = \
                    Viewer.__get_client(start_if_needed)
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
                        "No browser available for display. "
                        "Please install one manually.")
                    return  # Skip waiting since there is nothing to wait for

            # Wait to finish loading
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
            if self is None:
                self = Viewer

            if self is Viewer:
                # NEVER closing backend if closing instances, even for the
                # parent. It will be closed at Python exit automatically.
                Viewer._backend_robot_names.clear()
                Viewer.detach_camera()
                if Viewer.backend == 'meshcat' and Viewer.is_alive():
                    Viewer._backend_obj.close()
                    _ProcessWrapper(Viewer._backend_obj.recorder.proc).kill()
                    Viewer._backend_proc.kill()
                Viewer._backend_obj = None
                Viewer._backend_proc = None
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
    def _delete_nodes_viewer(nodes_path: Sequence[str]) -> None:
        """Delete a 'node' in Gepetto-viewer.

        .. note::
            Be careful, one must specify the full path of a node, including all
            parent group, but without the window name, ie
            'scene_name/robot_name' to delete the robot.

        :param nodes_path: Full path of the node to delete
        """
        if Viewer.backend.startswith('gepetto'):
            for node_path in nodes_path:
                Viewer._backend_obj.gui.deleteNode(node_path, True)
        else:
            for node_path in nodes_path:
                Viewer._backend_obj.gui[node_path].delete()

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
            H_orig = SE3(np.eye(3), body_transform.position)

        # Perform the desired rotation
        if Viewer.backend.startswith('gepetto'):
            H_abs = SE3(rotation_mat, position)
            if relative is None:
                self._gui.setCameraTransform(
                    self._client.windowID, SE3ToXYZQUAT(H_abs).tolist())
            else:
                # Not using recursive call for efficiency
                H_abs = H_orig * H_abs
                position = H_abs.position
                self._gui.setCameraTransform(
                    self._client.windowID, SE3ToXYZQUAT(H_abs).tolist())
        elif Viewer.backend.startswith('meshcat'):
            if relative is None:
                # Meshcat camera is rotated by -pi/2 along Roll axis wrt the
                # usual convention in robotics.
                position_meshcat = CAMERA_INV_TRANSFORM_MESHCAT @ position
                rotation_meshcat = matrixToRpy(
                    CAMERA_INV_TRANSFORM_MESHCAT @ rotation_mat)
                self._gui["/Cameras/default/rotated/<object>"].\
                    set_transform(mtf.compose_matrix(
                        translate=position_meshcat, angles=rotation_meshcat))
            else:
                H_abs = SE3(rotation_mat, position)
                H_abs = H_orig * H_abs
                position = H_abs.position
                return self.set_camera_transform(position, rotation)

        # Backup updated camera pose
        Viewer._camera_xyzrpy[0] = position.copy()
        Viewer._camera_xyzrpy[1] = rotation.copy()

    @staticmethod
    def add_camera_motion(camera_motion: CameraMotionType) -> None:
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
            camera_xyzrpy = DEFAULT_CAMERA_REL_XYZRPY

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
                rgba_array = np.array(img_obj)
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
                   width: int = DEFAULT_CAPTURE_SIZE,
                   height: int = DEFAULT_CAPTURE_SIZE) -> None:
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
        if Viewer.backend.startswith('gepetto'):
            self._gui.captureFrame(self._client.windowID, image_path)
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

            if Viewer.backend.startswith('gepetto'):
                for model, data, model_type in zip(
                        model_list, data_list, model_type_list):
                    self._gui.applyConfigurations(
                        [self._client.getViewerNodeName(geometry, model_type)
                            for geometry in model.geometryObjects],
                        [pin.SE3ToXYZQUATtuple(data.oMg[
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

            # Update the camera placement if necessary
            if Viewer._camera_travelling is not None:
                if Viewer._camera_travelling['viewer'] is self:
                    self.set_camera_transform(
                        *Viewer._camera_travelling['pose'],
                        relative=Viewer._camera_travelling['frame'])
            elif Viewer._camera_motion is not None:
                self.set_camera_transform()

            # Refreshing viewer backend manually is necessary for gepetto-gui
            if Viewer.backend.startswith('gepetto'):
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
        pin.updateGeometryPlacements(
            self._client.model, self._client.data,
            self._client.collision_model, self._client.collision_data)
        pin.framesForwardKinematics(self._client.model, self._client.data, q)

        # Refresh the viewer
        self.refresh(wait)

    def replay(self,
               evolution_robot: Sequence[State],
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
            if Viewer._camera_motion is not None:
                Viewer._camera_xyzrpy = Viewer._camera_motion(s.t)
            self.display(s.q, xyz_offset, wait)
            t_simu = (time.time() - init_time) * replay_speed
            i = bisect_right(t, t_simu)
            sleep(s.t - t_simu)
            wait = False  # It is enough to wait for the first timestep only


class TrajectoryDataType(TypedDict, total=False):
    # List of State objects of increasing time.
    evolution_robot: Sequence[State]
    # Jiminy robot. None if omitted.
    robot: Optional[jiminy.Robot]
    # Whether to use theoretical or actual model
    use_theoretical_model: bool


def extract_viewer_data_from_log(log_data: Dict[str, np.ndarray],
                                 robot: jiminy.Robot) -> TrajectoryDataType:
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


ColorType = Union[Tuple4FType, str]


def play_trajectories(trajectory_data: Union[
                          TrajectoryDataType, Sequence[TrajectoryDataType]],
                      replay_speed: float = 1.0,
                      record_video_path: Optional[str] = None,
                      viewers: Sequence[Viewer] = None,
                      start_paused: bool = False,
                      wait_for_client: bool = True,
                      travelling_frame: Optional[str] = None,
                      camera_xyzrpy: Optional[CameraPoseType] = None,
                      camera_motion: Optional[CameraMotionType] = None,
                      xyz_offset: Optional[Union[
                          Tuple3FType, Sequence[Tuple3FType]]] = None,
                      urdf_rgba: Optional[Union[
                          ColorType, Sequence[ColorType]]] = None,
                      backend: Optional[str] = None,
                      window_name: str = 'jiminy',
                      scene_name: str = 'world',
                      close_backend: Optional[bool] = None,
                      delete_robot_on_close: Optional[bool] = None,
                      legend: Optional[Union[str, Sequence[str]]] = None,
                      watermark_fullpath: Optional[str] = None,
                      verbose: bool = True) -> Sequence[Viewer]:
    """Replay one or several robot trajectories in a viewer.

    The ratio between the replay and the simulation time is kept constant to
    the desired ratio. One can choose between several backend (gepetto-gui or
    meshcat).

    .. note::
        Replay speed is independent of the platform (windows, linux...) and
        available CPU power.

    :param trajectory_data: List of `TrajectoryDataType` dicts.
    :param replay_speed: Speed ratio of the simulation.
                         Optional: 1.0 by default.
    :param record_video_path: Fullpath location where to save generated video
                              (.mp4 extension is: mandatory). Must be specified
                              to enable video recording. None to disable.
                              Optional: None by default.
    :param viewers: List of already instantiated viewers, associated one by one
                    in order to each trajectory data. None to disable.
                    Optional: None by default.
    :param start_paused: Start the simulation is pause, waiting for keyboard
                         input before starting to play the trajectories.
                         Optional: False by default.
    :param wait_for_client: Wait for the client to finish loading the meshes
                            before starting.
                            Optional: True by default.
    :param travelling_frame: Name of the frame of the robot associated with the
                             first trajectory_data. The camera will
                             automatically follow it. None to disable.
                             Optional: None by default.
    :param camera_xyzrpy: Tuple position [X, Y, Z], rotation [Roll, Pitch, Yaw]
                          corresponding to the absolute pose of the camera
                          during replay, if travelling is disable, or the
                          relative pose wrt the tracked frame otherwise. None
                          to disable.
                          Optional: None by default.
    :param camera_motion: Camera breakpoint poses over time, as a list of
                          `CameraMotionBreakpointType` dict. None to disable.
                          Optional: None by default.
    :param xyz_offset: List of constant position of the root joint for each
                       robot in world frame. None to disable.
                       Optional: None by default.
    :param urdf_rgba: List of RGBA code defining the color for each robot. It
                      will apply to every link. None to disable.
                      Optional: Original color if single robot, default color
                      cycle otherwise.
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
    :param legend: List of text defining the legend for each robot. `urdf_rgba`
                   must be specified to enable this option. It is not
                   persistent but disabled after replay. This option is only
                   supported by meshcat backend. None to disable.
                   Optional: No legend if no color by default, the robots names
                   otherwise.
    :param watermark_fullpath: Add watermark to the viewer. It is not
                               persistent but disabled after replay. This
                               option is only supported by meshcat backend.
                               None to disable.
                               Optional: No watermark by default.
    :param verbose: Add information to keep track of the process.
                    Optional: True by default.

    :returns: List of viewers used to play the trajectories.
    """
    if not isinstance(trajectory_data, (list, tuple)):
        trajectory_data = [trajectory_data]

    # Sanitize user-specified viewers
    if viewers is not None:
        # Make sure that viewers is a list
        if not isinstance(viewers, (list, tuple)):
            viewers = [viewers]

        # Make sure the viewers are still running if specified
        if not Viewer.is_open():
            viewers = None
        else:
            for viewer in viewers:
                if viewer is None or not viewer.is_open():
                    viewers = None
                    break

        # Do not close backend by default if it was supposed to be available
        if close_backend is None:
            close_backend = False

    # Sanitize user-specified robot offsets
    if xyz_offset is None:
        xyz_offset = len(trajectory_data) * [None]
    assert len(xyz_offset) == len(trajectory_data)

    # Sanitize user-specified robot colors
    if urdf_rgba is None:
        if len(trajectory_data) == 1:
            urdf_rgba = [None]
        else:
            urdf_rgba = list(islice(
                cycle(DEFAULT_URDF_COLORS.values()), len(trajectory_data)))
    elif not isinstance(urdf_rgba, (list, tuple)) or \
            isinstance(urdf_rgba[0], float):
        urdf_rgba = [urdf_rgba]
    elif isinstance(urdf_rgba, tuple):
        urdf_rgba = list(urdf_rgba)
    for i, color in enumerate(urdf_rgba):
        if isinstance(color, str):
            urdf_rgba[i] = DEFAULT_URDF_COLORS[color]
    assert len(urdf_rgba) == len(trajectory_data)

    # Sanitize user-specified legend
    if legend is not None and not isinstance(legend, (list, tuple)):
        legend = [legend]
    if all(color is not None for color in urdf_rgba):
        if legend is None:
            legend = [viewer.robot_name for viewer in viewers]
    else:
        legend = None
        logging.warning(
            "Impossible to display legend if at least one URDF do not "
            "have custom color.")

    # Instantiate or refresh viewers if necessary
    if viewers is None:
        # Delete robot by default only if not in notebook
        if delete_robot_on_close is None:
            delete_robot_on_close = not interactive_mode()

        # Create new viewer instances
        viewers = []
        lock = Lock()
        uniq_id = next(tempfile._get_candidate_names())
        for i, (traj, color) in enumerate(zip(trajectory_data, urdf_rgba)):
            # Create a new viewer instance, and load the robot in it
            robot = traj['robot']
            robot_name = f"{uniq_id}_robot_{i}"
            use_theoretical_model = traj['use_theoretical_model']
            viewer = Viewer(
                robot,
                use_theoretical_model=use_theoretical_model,
                urdf_rgba=color,
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
    else:
        # Reset robot model in viewer if requested color has changed
        for viewer, traj, color in zip(viewers, trajectory_data, urdf_rgba):
            if color != viewer.urdf_rgba:
                viewer._setup(traj['robot'], color)
    assert len(viewers) == len(trajectory_data)

    # # Early return if nothing to replay
    if all(not len(traj['evolution_robot']) for traj in trajectory_data):
        return viewers

    # Set camera pose or activate camera travelling if requested
    if travelling_frame is not None:
        viewers[0].attach_camera(travelling_frame, camera_xyzrpy)
    elif camera_xyzrpy is not None:
        viewers[0].set_camera_transform(*camera_xyzrpy)

    # Enable camera motion if requested
    if camera_motion is not None:
        Viewer.add_camera_motion(camera_motion)

    # Handle meshcat-specific options
    if Viewer.backend == 'meshcat':
        if legend is not None:
            assert len(legend) == len(trajectory_data)
            for viewer, color, text in zip(viewers, urdf_rgba, legend):
                rgba = [*[int(e * 255) for e in color[:3]], color[3]]
                color = f"rgba({','.join(map(str, rgba))}"
                Viewer._backend_obj.set_legend_item(
                    viewer.robot_name, color, text)

        # Add watermark if requested
        if watermark_fullpath is not None:
            Viewer._backend_obj.set_watermark(
                watermark_fullpath, *DEFAULT_WATERMARK_MAXSIZE)

    # Load robots in gepetto viewer
    for viewer, traj, offset in zip(viewers, trajectory_data, xyz_offset):
        if len(traj['evolution_robot']):
            viewer.display(traj['evolution_robot'][0].q, offset)

    # Wait for the meshes to finish loading if non video recording mode
    if wait_for_client and record_video_path is None:
        if Viewer.backend.startswith('meshcat'):
            if verbose and not interactive_mode():
                print("Waiting for meshcat client in browser to connect: "
                      f"{Viewer._backend_obj.gui.url()}")
            Viewer.wait(require_client=True)
            if verbose and not interactive_mode():
                print("Browser connected! Starting to replay the simulation.")

    # Handle start-in-pause mode
    if start_paused and not interactive_mode():
        input("Press Enter to continue...")

    # Replay the trajectory
    if record_video_path is not None:
        # Extract and resample trajectory data at fixed framerate
        time_max = 0.0
        for traj in trajectory_data:
            if len(traj['evolution_robot']):
                time_max = max([time_max, traj['evolution_robot'][-1].t])

        time_evolution = np.arange(
            0.0, time_max, replay_speed / VIDEO_FRAMERATE)
        position_evolutions = []
        for traj in trajectory_data:
            if len(traj['evolution_robot']):
                data_orig = traj['evolution_robot']
                t_orig = np.array([s.t for s in data_orig])
                pos_orig = np.stack([s.q for s in data_orig], axis=0)
                pos_interp = interp1d(
                    t_orig, pos_orig,
                    kind='linear', bounds_error=False,
                    fill_value=(pos_orig[0], pos_orig[-1]), axis=0)
                position_evolutions.append(pos_interp(time_evolution))
            else:
                position_evolutions.append(None)

        # Play trajectories without multithreading and record_video
        is_initialized = False
        for i in tqdm(range(len(time_evolution)),
                      desc="Rendering frames",
                      disable=(not verbose)):
            for viewer, positions, offset in zip(
                    viewers, position_evolutions, xyz_offset):
                if positions is not None:
                    viewer.display(
                        positions[i], xyz_offset=offset)
            if Viewer.backend != 'meshcat':
                import cv2
                frame = viewers[0].capture_frame(VIDEO_SIZE[1], VIDEO_SIZE[0])
                if not is_initialized:
                    record_video_path = str(
                        pathlib.Path(record_video_path).with_suffix('.mp4'))
                    out = cv2.VideoWriter(
                        record_video_path, cv2.VideoWriter_fourcc(*'vp09'),
                        fps=VIDEO_FRAMERATE, frameSize=frame.shape[1::-1])
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                if not is_initialized:
                    viewers[0]._backend_obj.start_recording(
                        VIDEO_FRAMERATE, *VIDEO_SIZE)
                viewers[0]._backend_obj.add_frame()
            is_initialized = True
        if Viewer.backend != 'meshcat':
            out.release()
        else:
            record_video_path = str(
                pathlib.Path(record_video_path).with_suffix('.webm'))
            viewers[0]._backend_obj.stop_recording(record_video_path)
    else:
        def replay_thread(viewer, *args):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            viewer.replay(*args)

        # Play trajectories with multithreading
        threads = []
        for viewer, traj, offset in zip(viewers, trajectory_data, xyz_offset):
            threads.append(Thread(
                target=replay_thread,
                args=(viewer,
                      traj['evolution_robot'],
                      replay_speed,
                      offset,
                      wait_for_client)))
        for thread in threads:
            thread.daemon = True
            thread.start()
        for thread in threads:
            thread.join()

    # Disable camera travelling and camera motion if it was enabled
    if travelling_frame is not None:
        Viewer.detach_camera()
    if camera_motion is not None:
        Viewer.remove_camera_motion()

    # Handle meshcat-specific options
    if Viewer.backend == 'meshcat':
        # Disable legend if it was enabled
        if legend is not None:
            for viewer in viewers:
                Viewer._backend_obj.remove_legend_item(viewer.robot_name)

        # Disable watermark if it was enabled
        if watermark_fullpath is not None:
            Viewer._backend_obj.remove_watermark()

    # Close backend if needed
    if close_backend:
        for viewer in viewers:
            viewer.close()

    return viewers


def play_logfiles(robots: Union[Sequence[jiminy.Robot], jiminy.Robot],
                  logs_data: Union[Sequence[Dict[str, np.ndarray]],
                                   Dict[str, np.ndarray]],
                  **kwargs) -> Sequence[Viewer]:
    """Play the content of a logfile in a viewer.

    This method simply formats the data then calls play_trajectories.

    :param robots: Either a single robot, or a list of robot for each log data.
    :param logs_data: Either a single dictionary, or a list of dictionaries of
                      simulation data log.
    :param kwargs: Keyword arguments to forward to `play_trajectories` method.
    """
    # Reformat everything as lists
    if not isinstance(logs_data, (list, tuple)):
        logs_data = [logs_data]
    if not isinstance(robots, (list, tuple)):
        robots = [robots] * len(logs_data)

    # For each pair (log, robot), extract a trajectory object for
    # `play_trajectories`
    trajectories = [extract_viewer_data_from_log(log, robot)
                    for log, robot in zip(logs_data, robots)]

    # Finally, play the trajectories
    return play_trajectories(trajectories, **kwargs)
