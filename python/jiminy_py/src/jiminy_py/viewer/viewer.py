import os
import re
import io
import sys
import time
import math
import shutil
import base64
import atexit
import logging
import pathlib
import tempfile
import subprocess
import webbrowser
import multiprocessing
from copy import deepcopy
from urllib.request import urlopen
from functools import wraps, partial
from bisect import bisect_right
from threading import RLock
from typing import Optional, Union, Sequence, Tuple, Dict, Callable, Any

import psutil
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from typing_extensions import TypedDict

import zmq
import meshcat.transformations as mtf
from panda3d_viewer.viewer_errors import ViewerClosedError

import pinocchio as pin
from pinocchio import SE3, SE3ToXYZQUAT
from pinocchio.rpy import rpyToMatrix, matrixToRpy

from .. import core as jiminy
from ..core import (ContactSensor as contact,
                    discretize_heightmap)
from ..state import State
from .meshcat.utilities import interactive_mode
from .meshcat.wrapper import MeshcatWrapper
from .meshcat.meshcat_visualizer import MeshcatVisualizer
from .panda3d.panda3d_visualizer import (Tuple3FType,
                                         Tuple4FType,
                                         Panda3dApp,
                                         Panda3dViewer,
                                         Panda3dVisualizer)


REPLAY_FRAMERATE = 40

CAMERA_INV_TRANSFORM_PANDA3D = rpyToMatrix(-np.pi/2, 0.0, 0.0)
CAMERA_INV_TRANSFORM_MESHCAT = rpyToMatrix(-np.pi/2, 0.0, 0.0)
DEFAULT_CAMERA_XYZRPY_ABS = ([7.5, 0.0, 1.4], [1.4, 0.0, np.pi/2])
DEFAULT_CAMERA_XYZRPY_REL = ([4.5, -4.5, 0.75], [1.3, 0.0, 0.8])

DEFAULT_WATERMARK_MAXSIZE = (150, 150)

# Fz force value corresponding to capsule's length of 1m
CONTACT_FORCE_SCALE = 10000.0  # [N]
EXTERNAL_FORCE_SCALE = 800.0  # [N]

COLORS = {'red': (0.9, 0.15, 0.15, 1.0),
          'blue': (0.3, 0.3, 1.0, 1.0),
          'green': (0.4, 0.7, 0.3, 1.0),
          'yellow': (1.0, 0.7, 0.0, 1.0),
          'purple': (0.6, 0.2, 0.9, 1.0),
          'orange': (1.0, 0.45, 0.0, 1.0),
          'grey': (0.55, 0.55, 0.55, 1.0),
          'cyan': (0.2, 0.7, 1.0, 1.0),
          'white': (1.0, 1.0, 1.0, 1.0),
          'black': (0.2, 0.2, 0.25, 1.0)}


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


def get_backends_available() -> Dict[str, type]:
    """Determine the set of available backends.
    """
    # In must be a function, otherwise it would only be run once at import by
    # the main thread only.
    backends_available = {'panda3d-sync': Panda3dVisualizer}
    if not multiprocessing.current_process().daemon:
        backends_available.update({'meshcat': MeshcatVisualizer,
                                   'panda3d': Panda3dVisualizer})
        try:
            from .panda3d.panda3d_widget import Panda3dQWidget  # noqa: F401
            backends_available['panda3d-qt'] = Panda3dVisualizer
        except ImportError:
            pass
    return backends_available


def check_display_available() -> bool:
    """Check if graphical server is available for onscreen rendering.
    """
    if multiprocessing.current_process().daemon:
        return False
    if not (sys.platform.startswith("win") or os.environ.get("DISPLAY")):
        return False
    return True


def get_default_backend() -> str:
    """Determine the default backend viewer, depending eventually on the
    running environment, hardware, and set of available backends.

    Meshcat will always be favored in interactive mode, i.e. in Jupyter
    notebooks, Panda3d otherwise. For Panda3d, synchronous mode without
    subprocess is preferred if onscreen display is impossible.

    .. note::
        Both Meshcat and Panda3d supports Nvidia EGL rendering without
        graphical server. Besides, both can fallback to software rendering if
        necessary, although Panda3d offers only very limited support of it.
    """
    if interactive_mode():
        return 'meshcat'
    elif check_display_available():
        return 'panda3d'
    else:
        return 'panda3d-sync'


def sleep(dt: float) -> None:
    """Function to provide cross-platform time sleep with maximum accuracy.

    .. warning::
        Use this method with cautious since it relies on busy looping principle
        instead of system scheduler. As a result, it wastes a lot more
        resources than time.sleep. However, it is the only way to ensure
        accurate delay on a non-real-time systems such as Windows 10.

    :param dt: Sleep duration in seconds.
    """
    # A new high-precision cross-platform sleep method is now available
    if sys.version_info >= (3, 11):
        time.sleep(dt)
        return

    # Estimate of timer jitter depending on the operating system
    if sys.platform.startswith('win'):
        timer_jitter = 1e-2
    else:
        timer_jitter = 1e-3

    # Combine busy loop and timer to release the GIL periodically
    t_end = time.perf_counter() + dt
    while time.perf_counter() < t_end:
        if t_end - time.perf_counter() > timer_jitter:
            time.sleep(1e-3)


def get_color_code(color: Optional[Union[str, Tuple4FType]]) -> Tuple4FType:
    if isinstance(color, str):
        try:
            return COLORS[color]
        except KeyError as e:
            colors_str = ', '.join(f"'{e}'" for e in COLORS.keys())
            raise ValueError(
                f"Color '{color}' not available. Use a custom (R,G,B,A) "
                f"code, or a predefined named color ({colors_str}).") from e
    return color


class _ProcessWrapper:
    """Wrap `multiprocessing.process.BaseProcess`, `subprocess.Popen`, and
    `psutil.Process` in the same object to provide the same user interface.

    It also makes sure that the process is properly terminated at Python exits,
    and without zombies left behind.
    """
    def __init__(self,
                 proc: Union[
                     multiprocessing.process.BaseProcess, subprocess.Popen,
                     psutil.Process, Panda3dApp],
                 kill_at_exit: bool = False):
        self._proc = proc
        # Make sure the process is killed at Python exit
        if kill_at_exit:
            atexit.register(self.kill)

    def is_parent(self) -> bool:
        return not isinstance(self._proc, psutil.Process)

    def is_alive(self) -> bool:
        if isinstance(self._proc, multiprocessing.process.BaseProcess):
            return self._proc.is_alive()
        elif isinstance(self._proc, subprocess.Popen):
            return self._proc.poll() is None
        elif isinstance(self._proc, psutil.Process):
            try:
                return self._proc.status() in [
                    psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]
            except psutil.NoSuchProcess:
                return False
        elif isinstance(self._proc, Panda3dApp):
            return hasattr(self._proc, 'win')
        return False  # Assuming it is not running by default

    def wait(self, timeout: Optional[float] = None) -> bool:
        if isinstance(self._proc, multiprocessing.process.BaseProcess):
            return self._proc.join(timeout)
        elif isinstance(self._proc, (
                subprocess.Popen, psutil.Process)):
            return self._proc.wait(timeout)
        elif isinstance(self._proc, Panda3dApp):
            self._proc.step()
            return True

    def kill(self) -> None:
        if self.is_parent() and self.is_alive():
            if isinstance(self._proc, Panda3dApp):
                self._proc.destroy()
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


class MarkerDataType(TypedDict, total=True):
    """Pose of the marker, as a single vector (position [X, Y, Z] + rotation
    [Quat X, Quat Y, Quat Z, Quat W]).
    """
    pose: np.ndarray
    """Size of the marker. Each principal axis of the geometry are scaled
    separately.
    """
    scale: np.ndarray
    """Color of the marker, as a list of 4 floating-point values ranging from
    0.0 to 1.0.
    """
    color: np.ndarray


class Viewer:
    """ TODO: Write documentation.

    .. note::
        The environment variable 'JIMINY_VIEWER_INTERACTIVE_DISABLE' can be
        used to force disabling interactive display.
    """
    backend = None
    window_name = 'jiminy'
    _has_gui = False
    _backend_obj = None
    _backend_proc = None
    _backend_robot_names = set()
    _backend_robot_colors = {}
    _camera_motion = None
    _camera_travelling = None
    _camera_xyzrpy = list(deepcopy(DEFAULT_CAMERA_XYZRPY_ABS))
    _lock = RLock()  # Unique lock for every viewer in same thread by default

    def __init__(self,
                 robot: jiminy.Model,
                 use_theoretical_model: bool = False,
                 robot_color: Optional[Union[str, Tuple4FType]] = None,
                 lock: Optional[RLock] = None,
                 backend: Optional[str] = None,
                 open_gui_if_parent: Optional[bool] = None,
                 delete_robot_on_close: bool = False,
                 robot_name: Optional[str] = None,
                 scene_name: str = 'world',
                 display_com: bool = False,
                 display_dcm: bool = False,
                 display_contact_frames: bool = False,
                 display_contact_forces: bool = False,
                 display_f_external: Optional[
                     Union[Sequence[bool], bool]] = None,
                 **kwargs: Any):
        """
        :param robot: Jiminy.Model to display.
        :param use_theoretical_model: Whether to use the theoretical (rigid)
                                      model or the actual (flexible) model of
                                      this robot. Note that using the actual
                                      model is more efficient since update of
                                      the frames placements can be skipped.
                                      Optional: Actual model by default.
        :param robot_color: Color of the robot. It will override the original
                            color of the meshes if not `None`. It supports both
                            RGBA codes as a list of 4 floating-point values
                            ranging from 0.0 and 1.0, and a few named colors.
                            Optional: Disabled by default.
        :param lock: Custom threading.RLock. Required for parallel rendering.
                     It is required since some backends does not support
                     multiple simultaneous connections (e.g. corbasever).
                     `None` to use the unique lock of the current thread.
                     Optional: `None` by default.
        :param backend: Name of the rendering backend to use. It can be either
                        'panda3d', 'panda3d-qt', 'meshcat'. None to keep using
                        to one already running if any, or the default one
                        otherwise. Note that the default is hardware and
                        environment dependent. See `viewer.default_backend`
                        method for details.
                        Optional: `None` by default.
        :param open_gui_if_parent: Open GUI if new viewer's backend server is
                                   started. `None` to fallback to default.
                                   Optional: Do not open gui for 'meshcat'
                                   backend in interactive mode with already one
                                   display cell already opened, open gui by
                                   default in any other case if graphical
                                   server is available.
        :param delete_robot_on_close: Enable automatic deletion of the robot
                                      when closing.
                                      Optional: False by default.
        :param robot_name: Unique robot name, to identify each robot.
                           Optional: Randomly generated identifier by default.
        :param scene_name: Scene name.
                           Optional: 'world' by default.
        :param display_com: Whether or not to display the center of mass.
                            Optional: Disabled by default.
        :param display_dcm: Whether or not to display the capture point / DCM.
                            Optional: Disabled by default.
        :param display_contact_frames:
            Whether or not to display the contact frames.
            Optional: Disabled by default.
        :param display_contact_forces:
            Whether or not to display the contact forces. Note that the user is
            responsible for updating sensors data since `Viewer.display` is
            only computing kinematic quantities.
            Optional: Disabled by default.
        :param display_f_external:
            Whether or not to display the external external forces applied at
            the joints on the robot. If a boolean is provided, the same
            visibility will be set for each joint, alternatively one can
            provide a boolean list whose ordering is consistent with
            `pinocchio_model.names`. Note that the user is responsible for
            updating the force buffer `viewer.f_external` data since
            `Viewer.display` is only computing kinematic quantities.
            Optional: Root joint for robot with freeflyer by default.
        :param kwargs: Unused extra keyword arguments to enable forwarding.
        """
        # Handling of default arguments
        if robot_name is None:
            uniq_id = next(tempfile._get_candidate_names())
            robot_name = "_".join(("robot", uniq_id))

        if backend is None:
            if Viewer.backend is not None:
                backend = Viewer.backend
            else:
                backend = get_default_backend()

        # Access the current backend or create one if none is available
        self.__is_open = False
        self.is_backend_parent = not Viewer.is_alive()
        try:
            # Start viewer backend
            Viewer.connect_backend(backend)

            # Decide whether or not to open gui
            if open_gui_if_parent is None:
                if not check_display_available():
                    open_gui_if_parent = False
                elif backend == 'meshcat':
                    # Opening a new display cell automatically if there is
                    # no other display cell already opened.
                    open_gui_if_parent = interactive_mode() and (
                        Viewer._backend_obj is None or
                        not Viewer._backend_obj.comm_manager.n_comm)
                elif backend == 'panda3d':
                    open_gui_if_parent = not interactive_mode()
                else:
                    open_gui_if_parent = False

            # Keep track of the backend process associated to the viewer.
            # The destructor of this instance must adapt its behavior to the
            # case where the backend process has changed in the meantime.
            self._gui = Viewer._backend_obj.gui
            self._backend_proc = Viewer._backend_proc
            self.__is_open = True

            # Open gui if requested
            try:
                if open_gui_if_parent:
                    Viewer.open_gui()
            except RuntimeError as e:
                # Convert exception into warning if it fails. It is probably
                # because no display is available.
                logger.warning(str(e))
        except Exception as e:
            self.close()
            raise RuntimeError(
                "Impossible to create backend or connect to it.") from e

        # Enforce some arguments based on available features
        if not Viewer.backend.startswith('panda3d'):
            if display_com or display_dcm or display_contact_frames or \
                    display_contact_forces:
                logger.warning(
                    "Panda3d backend is required to display markers, e.g. "
                    "CoM, DCM or Contact.")
            display_com = False
            display_dcm = False
            display_contact_frames = False
            display_contact_forces = False

        # Backup some user arguments
        self.robot_color = get_color_code(robot_color)
        self.robot_name = robot_name
        self.scene_name = scene_name
        self.use_theoretical_model = use_theoretical_model
        self.delete_robot_on_close = delete_robot_on_close
        self._lock = lock or Viewer._lock
        self._display_com = display_com
        self._display_dcm = display_dcm
        self._display_contact_frames = display_contact_frames
        self._display_contact_forces = display_contact_forces
        self._display_f_external = display_f_external

        # Initialize marker register
        self.markers: Dict[str, MarkerDataType] = {}
        self._markers_group = '/'.join((
            self.scene_name, self.robot_name, "markers"))
        self._markers_visibility: Dict[str, bool] = {}

        # Initialize external forces
        if self.use_theoretical_model:
            pinocchio_model = robot.pinocchio_model_th
        else:
            pinocchio_model = robot.pinocchio_model
        self.f_external = pin.StdVec_Force()
        self.f_external.extend([
            pin.Force.Zero() for _ in range(pinocchio_model.njoints - 1)])

        # Make sure that the windows, scene and robot names are valid
        if scene_name == Viewer.window_name:
            raise ValueError(
                "The name of the scene and window must be different.")

        if robot_name in Viewer._backend_robot_names:
            raise ValueError(
                "Robot name already exists but must be unique. Please choose "
                "a different one, or close the associated viewer.")

        # Create a unique temporary directory, specific to this viewer instance
        self._tempdir = tempfile.mkdtemp(
            prefix="_".join((Viewer.window_name, scene_name, robot_name, "")))

        # Load the robot
        self._setup(robot, self.robot_color)
        Viewer._backend_robot_names.add(self.robot_name)
        Viewer._backend_robot_colors.update({
            self.robot_name: self.robot_color})

        # Set default camera pose
        if self.is_backend_parent:
            self.set_camera_transform()

        # Refresh the viewer since the positions of the meshes and their
        # visibility mode are not properly set at this point.
        self.refresh(
            force_update_visual=True, force_update_collision=True, wait=True)

    def __del__(self) -> None:
        """Destructor.

        .. note::
            It automatically close the viewer before being garbage collected.
        """
        self.close()

    def __must_be_open(fct: Callable) -> Callable:
        @wraps(fct)
        def fct_safe(*args: Any, **kwargs: Any) -> Any:
            self = Viewer
            if args and isinstance(args[0], Viewer):
                self = args[0]
            self = kwargs.get('self', self)
            if not self.is_open():
                raise RuntimeError(
                    "No backend available. Please start one before calling "
                    f"'{fct.__name__}'.")
            return fct(*args, **kwargs)
        return fct_safe

    def __with_lock(fct: Callable) -> Callable:
        @wraps(fct)
        def fct_safe(*args: Any, **kwargs: Any) -> Any:
            self = Viewer
            if args and isinstance(args[0], Viewer):
                self = args[0]
            self = kwargs.get('self', self)
            with self._lock:
                return fct(*args, **kwargs)
        return fct_safe

    @__must_be_open
    @__with_lock
    def _setup(self,
               robot: jiminy.Model,
               robot_color: Optional[Union[str, Tuple4FType]] = None) -> None:
        """Load (or reload) robot in viewer.

        .. note::
            This method must be called after calling `engine.reset` since at
            this point the viewer has dangling references to the collision
            model and data of robot. Indeed, a new robot is generated at each
            reset to add some stockasticity to the mass distribution and some
            other parameters. This is done automatically if  one is using
            `simulator.Simulator` instead of `jiminy_py.core.Engine` directly.

        :param robot: jiminy.Model to display.
        :param robot_color: Color of the robot. It will override the original
                            color of the meshes if not `None`. It supports both
                            RGBA codes as a list of 4 floating-point values
                            ranging from 0.0 and 1.0, and a few named colors.
                            Optional: Disabled by default.
        """
        # Delete existing robot, if any
        robot_node_path = '/'.join((self.scene_name, self.robot_name))
        Viewer._delete_nodes_viewer([
            '/'.join((robot_node_path, "visuals")),
            '/'.join((robot_node_path, "collisions"))])

        # Backup desired color
        self.robot_color = get_color_code(robot_color)

        # Extract the right Pinocchio model
        if self.use_theoretical_model:
            pinocchio_model = robot.pinocchio_model_th
            pinocchio_data = robot.pinocchio_data_th
        else:
            pinocchio_model = robot.pinocchio_model
            pinocchio_data = robot.pinocchio_data

        # Reset external force buffer iif it is necessary
        if len(self.f_external) != pinocchio_model.njoints - 1:
            self.f_external = pin.StdVec_Force()
            self.f_external.extend([
                pin.Force.Zero() for _ in range(pinocchio_model.njoints - 1)])

        # Create backend wrapper to get (almost) backend-independent API
        self._client = get_backends_available()[Viewer.backend](
            pinocchio_model, robot.collision_model, robot.visual_model)
        self._client.data = pinocchio_data
        self._client.collision_data = robot.collision_data

        # Initialize the viewer
        self._client.initViewer(viewer=self._gui, loadModel=False)

        # Create the scene and load robot
        self._client.loadViewerModel(
            rootNodeName=robot_node_path, color=self.robot_color)

        if Viewer.backend.startswith('panda3d'):
            # Add markers' display groups
            self._gui.append_group(self._markers_group, remove_if_exists=False)

            # Extract data for fast access
            com_position = self._client.data.com[0]
            com_velocity = self._client.data.vcom[0]
            gravity = self._client.model.gravity.linear
            dcm = np.zeros(3)

            # Add center of mass
            def get_com_scale() -> Tuple3FType:
                nonlocal com_position
                return (1.0, 1.0, com_position[2])

            self.add_marker(name="COM_0_sphere",
                            shape="sphere",
                            pose=[com_position, None],
                            remove_if_exists=True,
                            auto_refresh=False,
                            radius=0.03)

            self.add_marker(name="COM_0_cylinder",
                            shape="cylinder",
                            pose=[com_position, None],
                            scale=get_com_scale,
                            remove_if_exists=True,
                            auto_refresh=False,
                            radius=0.004,
                            length=1.0,
                            anchor_bottom=True)

            self.display_center_of_mass(self._display_com)

            # Add DCM marker
            def get_dcm_pose() -> Tuple[Tuple3FType, Tuple4FType]:
                nonlocal com_position, com_velocity, gravity, dcm
                if com_position[2] > 0.0:
                    omega = math.sqrt(abs(gravity[2]) / com_position[2])
                    dcm[:2] = com_position[:2] + com_velocity[:2] / omega
                return (dcm, pin.Quaternion.Identity().coeffs())

            def get_dcm_scale() -> Tuple3FType:
                nonlocal com_position
                return np.full((3,), com_position[2] > 0.0, dtype=np.float64)

            self.add_marker(name="DCM",
                            shape="cone",
                            color="green",
                            pose=get_dcm_pose,
                            scale=get_dcm_scale,
                            remove_if_exists=True,
                            auto_refresh=False,
                            radius=0.03,
                            length=0.03,
                            num_sides=4)

            self.display_capture_point(self._display_dcm)

            # Add contact frame markers
            for frame_name, frame_idx in zip(
                    robot.contact_frames_names, robot.contact_frames_idx):
                frame_pose = self._client.data.oMf[frame_idx]
                self.add_marker(name='_'.join(("ContactFrame", frame_name)),
                                shape="sphere",
                                color="yellow",
                                pose=[frame_pose.translation, None],
                                remove_if_exists=True,
                                auto_refresh=False,
                                radius=0.02)

            self.display_contact_frames(self._display_contact_frames)

            # Add contact sensor markers
            def get_contact_scale(sensor_data: contact) -> Tuple3FType:
                f_z = - sensor_data[2]
                length = min(max(f_z / CONTACT_FORCE_SCALE, -1.0), 1.0)
                return (1.0, 1.0, length)

            if contact.type in robot.sensors_names.keys():
                for name in robot.sensors_names[contact.type]:
                    sensor = robot.get_sensor(contact.type, name)
                    frame_idx, data = sensor.frame_idx, sensor.data
                    self.add_marker(name='_'.join((contact.type, name)),
                                    shape="cylinder",
                                    pose=self._client.data.oMf[frame_idx],
                                    scale=partial(get_contact_scale, data),
                                    remove_if_exists=True,
                                    auto_refresh=False,
                                    radius=0.02,
                                    length=0.5,
                                    anchor_bottom=True)

            self.display_contact_forces(self._display_contact_forces)

            # Add external forces
            def get_force_pose(joint_idx: int,
                               joint_position: np.ndarray,
                               joint_rotation: np.ndarray
                               ) -> Tuple[Tuple3FType, Tuple4FType]:
                f = self.f_external[joint_idx - 1].linear
                f_rotation_local = pin.Quaternion(np.array([0.0, 0.0, 1.0]), f)
                f_rotation_world = (
                    pin.Quaternion(joint_rotation) * f_rotation_local).coeffs()
                return (joint_position, f_rotation_world)

            def get_force_scale(joint_idx: int) -> Tuple[float, float, float]:
                f_ext = self.f_external[joint_idx - 1].linear
                f_ext_norm = np.linalg.norm(f_ext, 2)
                length = min(f_ext_norm / EXTERNAL_FORCE_SCALE, 1.0)
                return (1.0, 1.0, length)

            for joint_name in pinocchio_model.names[1:]:
                joint_idx = self._client.model.getJointId(joint_name)
                joint_pose = self._client.data.oMi[joint_idx]
                pose_fn = partial(get_force_pose,
                                  joint_idx,
                                  joint_pose.translation,
                                  joint_pose.rotation)
                self.add_marker(name=f"ForceExternal_{joint_name}",
                                shape="arrow",
                                color="red",
                                pose=pose_fn,
                                scale=partial(get_force_scale, joint_idx),
                                remove_if_exists=True,
                                auto_refresh=False,
                                radius=0.015,
                                length=0.7)

            # Check if external forces visibility is deprecated
            njoints = pinocchio_model.njoints
            if isinstance(self._display_f_external, (list, tuple)):
                if len(self._display_f_external) != njoints - 1:
                    self._display_f_external = None

            # Display external forces only on freeflyer by default
            if self._display_f_external is None:
                self._display_f_external = \
                    [robot.has_freeflyer] + [False] * (njoints - 2)

            self.display_external_forces(self._display_f_external)

    @staticmethod
    @__must_be_open
    @__with_lock
    def open_gui() -> bool:
        """Open a new viewer graphical interface. It is only possible if a
        backend is already running.

        .. note::
            Only one graphical interface can be opened locally for efficiency.
        """
        # If a graphical window is already open, do nothing
        if Viewer.has_gui():
            return

        if Viewer.backend in ('panda3d-qt', 'panda3d-sync'):
            # No instance is considered manager of the unique window
            raise RuntimeError(
                f"Impossible to open gui with '{Viewer.backend}' backend.")
        elif Viewer.backend == 'panda3d':
            Viewer._backend_obj.gui.open_window()
        elif Viewer.backend == 'meshcat':
            viewer_url = Viewer._backend_obj.gui.url()

            if interactive_mode():
                from IPython.core.display import HTML, display

                # Scrap the viewer html content, including javascript
                # dependencies
                html_content = urlopen(viewer_url).read().decode()
                pattern = '<script type="text/javascript" src="%s"></script>'
                scripts_js = re.findall(pattern % '(.*)', html_content)
                for file in scripts_js:
                    file_path = os.path.join(viewer_url, file)
                    js_content = urlopen(file_path).read().decode()
                    html_content = html_content.replace(pattern % file, f"""
                    <script type="text/javascript">
                    {js_content}
                    </script>""")

                # Provide websocket URL as fallback if needed. It would be
                # the case if the environment is not jupyter-notebook nor
                # colab but rather jupyterlab or vscode for instance.
                from IPython import get_ipython
                from notebook import notebookapp
                kernel = get_ipython().kernel
                conn_file = kernel.config['IPKernelApp']['connection_file']
                kernel_id = conn_file.split('-', 1)[1].split('.')[0]
                server_pid = psutil.Process(os.getpid()).parent().pid
                server_list = list(notebookapp.list_running_servers())
                try:
                    from jupyter_server import serverapp
                    server_list += list(serverapp.list_running_servers())
                except ImportError:
                    pass
                for server_info in server_list:
                    if server_info['pid'] == server_pid:
                        break
                ws_path = (f"{server_info['base_url']}api/kernels/{kernel_id}"
                           f"/channels?token={server_info['token']}")
                html_content = html_content.replace(
                    "var ws_path = undefined;", f'var ws_path = "{ws_path}";')

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
    def has_gui() -> bool:
        if Viewer.is_alive():
            # Make sure the viewer still has gui if necessary
            if Viewer.backend == 'meshcat':
                comm_manager = Viewer._backend_obj.comm_manager
                if comm_manager is not None:
                    ack = Viewer._backend_obj.wait(require_client=False)
                    Viewer._has_gui = any(
                        msg == "ok" for msg in ack.split(","))
            return Viewer._has_gui
        return False

    @staticmethod
    @__must_be_open
    @__with_lock
    def wait(require_client: bool = False) -> None:
        """Wait for all the meshes to finish loading in every clients.

        :param require_client: Wait for at least one client to be available
                               before checking for mesh loading.
        """
        if Viewer.backend == 'meshcat':
            Viewer._backend_obj.wait(require_client)
        elif Viewer.backend.startswith('panda3d'):
            Viewer._backend_obj.gui.step()

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

    @__with_lock
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
        if self is None:
            self = Viewer

        if self is Viewer:
            # NEVER closing backend automatically if closing instances,
            # even for the parent. It will be closed at Python exit
            # automatically. One must call `Viewer.close` to do otherwise.
            Viewer._backend_robot_names.clear()
            Viewer._backend_robot_colors.clear()
            Viewer._camera_xyzrpy = list(
                deepcopy(DEFAULT_CAMERA_XYZRPY_ABS))
            Viewer.detach_camera()
            Viewer.remove_camera_motion()
            if Viewer.is_alive():
                if Viewer.backend == 'meshcat':
                    Viewer._backend_obj.close()
                elif Viewer.backend.startswith('panda3d'):
                    try:
                        Viewer._backend_obj.stop()
                    except ViewerClosedError:
                        pass
                Viewer._backend_proc.kill()
            atexit.unregister(Viewer.close)
            Viewer.backend = None
            Viewer._backend_obj = None
            Viewer._backend_proc = None
            Viewer._has_gui = False
        else:
            # Disable travelling if associated with this viewer instance
            if (Viewer._camera_travelling is not None and
                    Viewer._camera_travelling['viewer'] is self):
                Viewer.detach_camera()

            # Check if the backend process has changed or the viewer instance
            # has already been closed, which may happend if it has been closed
            # manually in the meantime. If so, there is nothing left to do.
            if Viewer._backend_proc is not self._backend_proc or \
                    not self.__is_open:
                return

            # Make sure zmq does not hang
            if Viewer.backend == 'meshcat' and Viewer.is_alive():
                Viewer._backend_obj.gui.window.zmq_socket.RCVTIMEO = 200

            # Consider that the robot name is now available, no matter
            # whether the robot has actually been deleted or not.
            Viewer._backend_robot_names.discard(self.robot_name)
            Viewer._backend_robot_colors.pop(self.robot_name)
            if self.delete_robot_on_close:
                Viewer._delete_nodes_viewer([
                    self._client.visual_group,
                    self._client.collision_group,
                    self._markers_group])

            # Restore zmq socket timeout, which is disable by default
            if Viewer.backend == 'meshcat':
                Viewer._backend_obj.gui.window.zmq_socket.RCVTIMEO = -1

            # Delete temporary directory
            if self._tempdir.startswith(tempfile.gettempdir()):
                try:
                    shutil.rmtree(self._tempdir)
                except FileNotFoundError:
                    pass

        # At this point, consider the viewer has been closed, no matter what
        self.__is_open = False

    @staticmethod
    @__with_lock
    def connect_backend(backend: Optional[str] = None,
                        close_at_exit: bool = True) -> None:
        """Get the running process of backend client.

        This method can be used to open a new process if necessary.

        :param backend: Name of the rendering backend to use. It can be either
                        'panda3d', 'panda3d-qt', 'meshcat'.
                        Optional: The default is hardware and environment
                        dependent. See `viewer.default_backend` for details.
        :param close_at_exit: Terminate backend server at Python exit.
                              Optional: True by default

        :returns: Pointer to the running backend Client and its PID.
        """
        # Handle default arguments
        if backend is None:
            backend = get_default_backend()

        # Sanitize user arguments
        backend = backend.lower()
        if backend not in get_backends_available():
            raise ValueError("%s backend not available." % backend)

        # Update the backend currently running, if any
        if Viewer.backend != backend and Viewer.is_alive():
            logging.warning("Different backend already running. Closing it...")
            Viewer.close()

        # Nothing to do if already connected
        if Viewer.is_alive():
            return

        # Reset some class attribute if backend not available
        Viewer.close()

        if backend.startswith('panda3d'):
            # Instantiate client with onscreen rendering capability enabled.
            # Note that it fallbacks to software rendering if necessary.
            if backend == 'panda3d-qt':
                from .panda3d.panda3d_widget import Panda3dQWidget
                client = Panda3dQWidget()
                proc = _ProcessWrapper(client, close_at_exit)
            elif backend == 'panda3d-sync':
                client = Panda3dApp()
                proc = _ProcessWrapper(client, close_at_exit)
            else:
                client = Panda3dViewer(window_type='onscreen',
                                       window_title=Viewer.window_name)
                proc = _ProcessWrapper(client._app, close_at_exit)

            # The gui is the client itself
            client.gui = client
        else:
            # List of connections likely to correspond to Meshcat servers
            meshcat_candidate_conn = {}
            for pid in psutil.pids():
                try:
                    proc = psutil.Process(pid)
                    for conn in proc.connections("tcp4"):
                        if conn.status != 'LISTEN' or \
                                conn.laddr.ip != '127.0.0.1':
                            continue
                        cmdline = proc.cmdline()
                        if cmdline and ('python' in cmdline[0].lower() or
                                        'meshcat' in cmdline[-1]):
                            meshcat_candidate_conn[pid] = conn
                except (psutil.AccessDenied,
                        psutil.ZombieProcess,
                        psutil.NoSuchProcess):
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
            for pid, conn in meshcat_candidate_conn.items():
                try:
                    # Note that the timeout must be long enough to give enough
                    # time to the server to respond, but not to long to avoid
                    # sending to much time spanning the available connections.
                    port = conn.laddr.port
                    if port in excluded_ports:
                        continue
                    zmq_url = f"tcp://127.0.0.1:{port}"
                    zmq_socket = context.socket(zmq.REQ)
                    zmq_socket.RCVTIMEO = 200  # millisecond
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
                proc = psutil.Process(pid)
            else:
                proc = client.server_proc
            proc = _ProcessWrapper(proc, close_at_exit)

        # Make sure the backend process is alive
        assert proc.is_alive(), (
            "Something went wrong. Impossible to instantiate viewer backend.")

        # Update global state
        Viewer.backend = backend
        Viewer._backend_obj = client
        Viewer._backend_proc = proc

        # Make sure to close cleanly the viewer at exit
        if close_at_exit:
            atexit.register(Viewer.close)

    @staticmethod
    @__must_be_open
    @__with_lock
    def _delete_nodes_viewer(nodes_path: Sequence[str]) -> None:
        """Delete an object or a group of objects in the scene.

        .. note::
            Be careful, one must specify the full path of a node, including all
            parent group, but without the window name, ie
            'scene_name/robot_name' to delete the robot.

        :param nodes_path: Full path of the node to delete
        """
        if Viewer.backend.startswith('panda3d'):
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
    @__with_lock
    def set_watermark(img_fullpath: Optional[str] = None,
                      width: Optional[int] = None,
                      height: Optional[int] = None) -> None:
        """Insert desired watermark on bottom left corner of the window.

        .. note::
            The relative width and height cannot exceed 20% of the visible
            area, otherwise it will be rescaled.

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
        if Viewer.backend.startswith('panda3d'):
            Viewer._backend_obj.gui.set_watermark(img_fullpath, width, height)
        else:
            width = width or DEFAULT_WATERMARK_MAXSIZE[0]
            height = height or DEFAULT_WATERMARK_MAXSIZE[1]
            if img_fullpath is None or img_fullpath == "":
                Viewer._backend_obj.remove_watermark()

    @staticmethod
    @__must_be_open
    @__with_lock
    def set_legend(labels: Optional[Sequence[str]] = None) -> None:
        """Insert legend on top left corner of the window.

        .. note::
            Make sure to have specified different colors for each robot on the
            scene, since it will be used as marker on the legend.

        :param labels: Sequence of strings whose length must be consistent
                       with the number of robots on the scene. None to disable.
                       Optional: None by default.
        """
        # Make sure number of labels is consistent with number of robots
        if labels is not None:
            assert len(labels) == len(Viewer._backend_robot_colors)

        if Viewer.backend.startswith('panda3d'):
            if labels is None:
                items = None
            else:
                items = list(zip(
                    labels, Viewer._backend_robot_colors.values()))
            Viewer._backend_obj.gui.set_legend(items)
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
    @__with_lock
    def set_clock(time: Optional[float] = None) -> None:
        """Insert clock on bottom right corner of the window.

        .. note::
            Only Panda3d rendering backend is supported by this method.

        :param time: Current time is seconds. None to disable.
                     Optional: None by default.
        """
        if Viewer.backend.startswith('panda3d'):
            Viewer._backend_obj.gui.set_clock(time)
        else:
            logger.warning("Adding clock is only available for Panda3d.")

    @__must_be_open
    @__with_lock
    def get_camera_transform(self) -> Tuple[Tuple3FType, Tuple3FType]:
        """Get transform of the camera pose.

        .. warning::
            The reference axis is negative z-axis instead of positive x-axis.

        .. warning::
            It returns the previous requested camera transform for meshcat,
            since it is impossible to get access to this information. Thus
            this method is valid as long as the user does not move the
            camera manually using mouse camera control.
        """
        if Viewer.backend.startswith('panda3d'):
            xyz, quat = self._gui.get_camera_transform()
            rot = pin.Quaternion(*quat).matrix()
            rpy = matrixToRpy(rot @ CAMERA_INV_TRANSFORM_PANDA3D.T)
        else:
            xyz, rpy = deepcopy(Viewer._camera_xyzrpy)
        return xyz, rpy

    @__must_be_open
    @__with_lock
    def set_camera_transform(self,
                             position: Optional[Tuple3FType] = None,
                             rotation: Optional[Tuple3FType] = None,
                             relative: Optional[Union[str, int]] = None,
                             wait: bool = False) -> None:
        """Set transform of the camera pose.

        .. warning::
            The reference axis is negative z-axis instead of positive x-axis,
            which means that position = [0.0, 0.0, 0.0], rotation =
            [0.0, 0.0, 0.0] moves the camera at the center of scene, looking
            downward.

        :param position: Position [X, Y, Z] as a list or 1D array. None to not
                         update it.
                         Optional: None by default.
        :param rotation: Rotation [Roll, Pitch, Yaw] as a list or 1D np.array.
                         None to note update it.
                         Optional: None by default.
        :param relative:
            .. raw:: html

                How to apply the transform:

            - **None:** absolute.
            - **'camera':** relative to current camera pose.
            - **other:** relative to a robot frame, not accounting for the
              rotation of the frame during travelling. It supports both frame
              name and index in model.
        :param wait: Whether or not to wait for rendering to finish.
        """
        # Handling of position and rotation arguments
        if position is None or rotation is None or relative == 'camera':
            position_camera, rotation_camera = self.get_camera_transform()
        if position is None:
            position = position_camera
        if rotation is None:
            rotation = rotation_camera
        position, rotation = np.asarray(position), np.asarray(rotation)

        # Compute associated rotation matrix
        rotation_mat = rpyToMatrix(rotation)

        # Compute the relative transformation if necessary
        if relative == 'camera':
            H_orig = SE3(rpyToMatrix(rotation_camera), position_camera)
        elif relative is not None:
            # Get the body position, not taking into account the rotation
            if isinstance(relative, str):
                relative = self._client.model.getFrameId(relative)
            try:
                body_transform = self._client.data.oMf[relative]
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
        if Viewer.backend.startswith('panda3d'):
            rotation_panda3d = pin.Quaternion(
                rotation_mat @ CAMERA_INV_TRANSFORM_PANDA3D).coeffs()
            self._gui.set_camera_transform(position, rotation_panda3d)
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
        Viewer._camera_xyzrpy = [position, rotation]

        # Wait for the backend viewer to finish rendering if requested
        if wait:
            Viewer.wait(require_client=False)

    @__must_be_open
    @__with_lock
    def set_camera_lookat(self,
                          position: Tuple3FType,
                          relative: Optional[Union[str, int]] = None,
                          wait: bool = False) -> None:
        """Set the camera look-at position.

        .. note::
            It preserve the relative camera pose wrt the lookup position.

        :param position: Position [X, Y, Z] as a list or 1D array, frame index
        :param relative: Set the lookat position relative to robot frame if
                         specified, in absolute otherwise. Both frame name and
                         index in model are supported.
        :param wait: Whether or not to wait for rendering to finish.
        """
        # Make sure the backend supports this method
        if not Viewer.backend.startswith('panda3d'):
            raise NotImplementedError(
                "This method is only supported by Panda3d.")

        # Compute absolute lookat position using frame and relative position
        if isinstance(relative, str):
            relative = self._client.model.getFrameId(relative)
        if isinstance(relative, int):
            body_transform = self._client.data.oMf[relative]
            position = body_transform.translation + position

        # Update camera lookat position
        self._gui.set_camera_lookat(position)

        # Wait for the backend viewer to finish rendering if requested
        if wait:
            Viewer.wait(require_client=False)

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
                      frame: Union[str, int],
                      camera_xyzrpy: Optional[CameraPoseType] = (None, None),
                      lock_relative_pose: Optional[bool] = None) -> None:
        """Attach the camera to a given robot frame.

        Only the position of the frame is taken into account. A custom relative
        pose of the camera wrt to the frame can be further specified. If so,
        then the relative camera pose wrt the frame is locked, otherwise the
        camera is only constrained to look at the frame.

        :param frame: Name or index of the frame of the robot to follow with
                      the camera.
        :param camera_xyzrpy: Tuple position [X, Y, Z], rotation
                              [Roll, Pitch, Yaw] corresponding to the relative
                              pose of the camera wrt the tracked frame. It will
                              be used to initialize the camera pose if relative
                              pose is not locked. `None` to disable.
                              Optional: Disabkle by default.
        :param lock_relative_pose: Whether or not to lock the relative pose of
                                   the camera wrt tracked frame.
                                   Optional: False by default iif Panda3d
                                   backend is used.
        """
        # Make sure one is not trying to track the camera itself
        assert frame != 'camera', "Impossible to track the camera itself !"

        # Make sure the frame exists and it is not the universe itself
        if isinstance(frame, str):
            frame = self._client.model.getFrameId(frame)
        if frame == self._client.model.nframes:
            raise ValueError("Trying to attach camera to non-existing frame.")
        assert frame != 0, "Impossible to track the universe !"

        # Handle of default camera lock mode
        if lock_relative_pose is None:
            lock_relative_pose = not Viewer.backend.startswith('panda3d')

        # Make sure camera lock mode is compatible with viewer backend
        if not lock_relative_pose and not Viewer.backend.startswith('panda3d'):
            raise NotImplementedError(
                "Not locking camera pose is only supported by Panda3d.")

        # Handling of default camera pose
        if lock_relative_pose and camera_xyzrpy is None:
            camera_xyzrpy = [None, None]

        # Set default relative camera pose if position/orientation undefined
        if camera_xyzrpy is not None:
            camera_xyzrpy = list(camera_xyzrpy)
            if camera_xyzrpy[0] is None:
                camera_xyzrpy[0] = deepcopy(DEFAULT_CAMERA_XYZRPY_REL[0])
            if camera_xyzrpy[1] is None:
                camera_xyzrpy[1] = deepcopy(DEFAULT_CAMERA_XYZRPY_REL[1])

        # Set camera pose if relative pose is not locked but provided
        if not lock_relative_pose and camera_xyzrpy is not None:
            self.set_camera_transform(*camera_xyzrpy, frame)
            camera_xyzrpy = None

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
    @__with_lock
    def set_color(self,
                  color: Optional[Union[str, Tuple4FType]] = None
                  ) -> None:
        """Override the color of the visual and collision geometries of the
        robot on-the-fly.

        .. note::
            This method is only supported by Panda3d for now.

        :param color: Color of the robot. It will override the original color
                      of the meshes if not `None`, and restore them otherwise.
                      It supports both RGBA codes as a list of 4 floating-point
                      values ranging from 0.0 and 1.0, and a few named colors.
                      Optional: Disabled by default.
        """
        # Sanitize user-specified color code
        color_ = get_color_code(color)

        if Viewer.backend.startswith('panda3d'):
            for model, geom_type in zip(
                    [self._client.visual_model, self._client.collision_model],
                    pin.GeometryType.names.values()):
                for geom in model.geometryObjects:
                    node_name = self._client.getViewerNodeName(geom, geom_type)
                    color = color_
                    if color is None and geom.overrideMaterial:
                        color = geom.meshColor
                    self._gui.set_material(
                        *node_name, color, disable_material=color_ is not None)
        else:
            logger.warning("This method is only supported by Panda3d.")

    @__must_be_open
    @__with_lock
    def update_floor(self,
                     ground_profile: Optional[jiminy.HeightmapFunctor] = None,
                     grid_size: float = 20.0,
                     grid_unit: float = 0.04,
                     show_meshes: bool = False) -> None:
        """Display a custom ground profile as a height map or the original tile
        ground floor.

        .. note::
            This method is only supported by Panda3d for now.

        :param ground_profile: `jiminy_py.core.HeightmapFunctor` associated
                               with the ground profile. It renders a flat tile
                               ground if not specified.
                               Optional: None by default.
        :param grid_size: X and Y dimension of the ground profile to render.
                          Optional: 20m by default.
        :param grid_unit: X and Y discretization step of the ground profile.
                          Optional: 4cm by default.
        :param show_meshes: Whether or not to highlight the meshes.
                            Optional: disabled by default.
        """
        if Viewer.backend.startswith('panda3d'):
            # Restore tile ground if heightmap is not specified
            if ground_profile is None:
                self._gui.update_floor()
                return

            # Discretize heightmap
            grid = discretize_heightmap(ground_profile, grid_size, grid_unit)

            # Make sure it is not flat ground
            if np.unique(grid[:, 2:], axis=0).shape[0] == 1 and \
                    np.allclose(grid[0, 2:], [0.0, 0.0, 0.0, 1.0], atol=1e-3):
                self._gui.update_floor()
                return

            self._gui.update_floor(grid, show_meshes)
        else:
            logger.warning("This method is only supported by Panda3d.")

    @__must_be_open
    @__with_lock
    def capture_frame(self,
                      width: int = None,
                      height: int = None,
                      raw_data: bool = False) -> Union[np.ndarray, bytes]:
        """Take a snapshot and return associated data.

        :param width: Width for the image in pixels. None to keep unchanged.
                      Optional: Kept unchanged by default.
        :param height: Height for the image in pixels. None to keep unchanged.
                       Optional: Kept unchanged by default.
        :param raw_data: Whether to return a 2D numpy array, or the raw output
                         from the backend (the actual type may vary).
        """
        # Check user arguments
        if Viewer.backend.startswith('panda3d'):
            # Resize window if size has changed
            _width, _height = self._gui.getSize()
            if width is None:
                width = _width
            if height is None:
                height = _height
            if _width != width or _height != height:
                self._gui.set_window_size(width, height)

            # Get raw buffer image instead of numpy array for efficiency
            buffer = self._gui.get_screenshot(requested_format='RGB', raw=True)

            # Return raw data if requested
            if raw_data:
                return buffer

            # Return numpy array RGB
            return np.frombuffer(buffer, np.uint8).reshape((height, width, 3))
        else:
            # Send capture frame request to the background recorder process
            img_html = Viewer._backend_obj.capture_frame(width, height)

            # Parse the output to remove the html header, and convert it into
            # the desired output format.
            img_data = str.encode(img_html.split(",", 1)[-1])
            buffer = base64.decodebytes(img_data)

            # Return raw data if requested
            if raw_data:
                return buffer

            # Return numpy array RGB
            with Image.open(io.BytesIO(buffer)) as img_obj:
                rgba_array = np.array(img_obj)
            return rgba_array[:, :, :-1]

    @__must_be_open
    @__with_lock
    def save_frame(self,
                   image_path: str,
                   width: int = None,
                   height: int = None) -> None:
        """Save a snapshot in png format.

        :param image_path: Fullpath of the image (.png extension is mandatory)
        :param width: Width for the image in pixels. None to keep unchanged.
                      Optional: Kept unchanged by default.
        :param height: Height for the image in pixels. None to keep unchanged.
                       Optional: Kept unchanged by default.
        """
        image_path = str(pathlib.Path(image_path).with_suffix('.png'))
        if Viewer.backend.startswith('panda3d'):
            _width, _height = self._gui.getSize()
            if width is None:
                width = _width
            if height is None:
                height = _height
            if _width != width or _height != height:
                self._gui.set_window_size(width, height)
            self._gui.save_screenshot(image_path)
        else:
            img_data = self.capture_frame(width, height, raw_data=True)
            with open(image_path, "wb") as f:
                f.write(img_data)

    @__must_be_open
    @__with_lock
    def display_visuals(self, visibility: bool) -> None:
        """Set the visibility of the visual model of the robot.

        :param visibility: Whether to enable or disable display of the visual
                           model.
        """
        self._client.displayVisuals(visibility)
        self.refresh()

    @__must_be_open
    @__with_lock
    def display_collisions(self, visibility: bool) -> None:
        """Set the visibility of the collision model of the robot.

        :param visibility: Whether to enable or disable display of the visual
                           model.
        """
        self._client.displayCollisions(visibility)
        self.refresh()

    @__must_be_open
    @__with_lock
    def add_marker(self,
                   name: str,
                   shape: str,
                   pose: Union[pin.SE3,
                               Sequence[Optional[np.ndarray]],
                               Callable[[], Tuple[Tuple3FType, Tuple4FType]]
                               ] = (None, None),
                   scale: Union[Union[float, Tuple3FType],
                                Callable[[], np.ndarray]] = 1.0,
                   color: Union[Optional[Union[str, Tuple4FType]],
                                Callable[[], Tuple4FType]] = None,
                   remove_if_exists: bool = False,
                   auto_refresh: bool = True,
                   *shape_args: Any,
                   **shape_kwargs: Any) -> MarkerDataType:
        """Add marker on the scene.

        .. warning::
            This method is only supported by Panda3d.

        :param name: Unique name. It must be a valid string identifier.
        :param shape: Desired shape, as a string, i.e. 'cone', 'box', 'sphere',
                      'capsule', 'cylinder', 'frame', or 'arrow'.
        :param pose: Pose of the geometry on the scene, as a transform object
                     `pin.SE3`, or a tuple (position, orientation). In the
                     latter case, the position must be the vector [X, Y, Z],
                     while the orientation can be either a rotation matrix, or
                     a quaternion [X, Y, Z, W]. `None` can be used to specify
                     neutral frame position and/or orientation.
                     Optional: Neutral position and orientation by default.
        :param scale: Size of the marker. Each principal axis of the geometry
                      are scaled separately.
        :param color: Color of the marker. It supports both RGBA codes as a
                      list of 4 floating-point values ranging from 0.0 and 1.0,
                      and a few named colors.
                      Optional: Robot's color by default if overridden,
                                'white' otherwise, except for 'frame'.
        :param auto_refresh: Whether or not to refresh the scene after adding
                             the marker. Useful for adding a bunch of markers
                             and only refresh once. Note that the marker will
                             not display properly until then.
        :param shape_args: Any additional positional arguments to forward to
                           `jiminy_py.viewer.panda3d.panda3d_visualizer.`
                           `Panda3dApp.append_{shape}`.
        :param shape_kwargs: Any additional keyword arguments to forward to
                             shape instantiation method.

        :returns: Dict of type `MarkerDataType`, storing references to the
                  current pose, scale, and color of the marker, and itself a
                  reference to `viewer.markers[name]`. Any modification of it
                  will take effect at next `refresh` call.
        """
        # Make sure the backend supports this method
        if not Viewer.backend.startswith('panda3d'):
            raise NotImplementedError(
                "This method is only supported by Panda3d.")

        # Handling of user arguments
        if pose is None:
            pose = [None, None]
        if isinstance(pose, pin.SE3):
            pose = [pose.translation, pose.rotation]
        if not callable(pose) and any(value is None for value in pose):
            pose = list(pose)
            if pose[0] is None:
                pose[0] = np.zeros(3)
            if pose[1] is None:
                pose[1] = np.array([0.0, 0.0, 0.0, 1.0])
        if np.isscalar(scale):
            scale = np.full((3,), fill_value=float(scale))
        if color is None:
            color = self.robot_color
        if color is None and shape != 'frame':
            color = 'white'
        if color is not None:
            color = np.asarray(get_color_code(color))

        # Remove marker is one already exists and requested
        if name in self.markers.keys():
            if not remove_if_exists:
                raise ValueError(f"marker's name '{name}' already exists.")
            self.remove_marker(name)

        # Add new marker
        create_shape = getattr(self._gui, f"append_{shape}")
        create_shape(self._markers_group, name, *shape_args, **shape_kwargs)
        marker_data = {"pose": pose, "scale": scale, "color": color}
        self.markers[name] = marker_data
        self._markers_visibility[name] = True

        # Make sure the marker always display in front of the model
        self._gui.show_node(
            self._markers_group, name, True, always_foreground=True)

        # Refresh the scene if desired
        if auto_refresh:
            self.refresh()

        return marker_data

    @__must_be_open
    @__with_lock
    def display_center_of_mass(self, visibility: bool) -> None:
        """Display the position of the center of mass as a sphere.

        .. note::
            It corresponds to the attribute `com[0]` of the provided
            `robot.pinocchio_model`. Calling `Viewer.display` will update it
            automatically, while `Viewer.refresh` will not.

        :param visibility: Whether to enable or disable display of the center
                           of mass.
        """
        if not Viewer.backend.startswith('panda3d'):
            raise NotImplementedError(
                "This method is only supported by Panda3d.")

        for name in self.markers:
            if name.startswith("COM_0"):
                self._gui.show_node(self._markers_group, name, visibility)
                self._markers_visibility[name] = visibility
        self._display_com = visibility

    @__must_be_open
    @__with_lock
    def display_capture_point(self, visibility: bool) -> None:
        """Display the position of the capture point,also called divergent
        component of motion (DCM) as a sphere.

        .. note::
            Calling `Viewer.display` will update it automatically, while
            `Viewer.refresh` will not.

        :param visibility: Whether to enable or disable display of the capture
                           point.
        """
        # Make sure the current backend is supported by this method
        if not Viewer.backend.startswith('panda3d'):
            raise NotImplementedError(
                "This method is only supported by Panda3d.")

        # Update visibility
        for name in self.markers:
            if name == "DCM":
                self._gui.show_node(self._markers_group, name, visibility)
                self._markers_visibility[name] = visibility
        self._display_dcm = visibility

        # Must refresh the scene
        if visibility:
            self.refresh()

    @__must_be_open
    @__with_lock
    def display_contact_frames(self, visibility: bool) -> None:
        """Display the contact frames of the robot as spheres.

        .. note::
            The frames to display are specified by the attribute
            `contact_frames_names` of the provided `robot`. Calling
            `Viewer.display` will update it automatically, while
            `Viewer.refresh` will not.

        .. warning::
            This method is only supported by Panda3d.

        :param visibility: Whether or not to display the contact frames.
        """
        # Make sure the current backend is supported by this method
        if not Viewer.backend.startswith('panda3d'):
            raise NotImplementedError(
                "This method is only supported by Panda3d.")

        # Update visibility
        for name in self.markers:
            if name.startswith("ContactFrame"):
                self._gui.show_node(self._markers_group, name, visibility)
                self._markers_visibility[name] = visibility
        self._display_contact_frames = visibility

        # Must refresh the scene
        if visibility:
            self.refresh()

    @__must_be_open
    @__with_lock
    def display_contact_forces(self, visibility: bool) -> None:
        """Display forces associated with the contact sensors attached to the
        robot, as cylinders of variable length depending of Fz.

        .. note::
            Fz can be signed. It will affect the orientation of the capsule.

        .. warning::
            It corresponds to the attribute `data` of `jiminy.ContactSensor`.
            Calling `Viewer.display` will NOT update its value automatically.
            It is up to the user to keep it up-to-date.

        .. warning::
            This method is only supported by Panda3d.

        :param visibility: Whether or not to display the contact forces.
        """
        # Make sure the current backend is supported by this method
        if not Viewer.backend.startswith('panda3d'):
            raise NotImplementedError(
                "This method is only supported by Panda3d.")

        # Update visibility
        for name in self.markers:
            if name.startswith(contact.type):
                self._gui.show_node(self._markers_group, name, visibility)
                self._markers_visibility[name] = visibility
        self._display_contact_forces = visibility

        # Must refresh the scene
        if visibility:
            self.refresh()

    @__must_be_open
    @__with_lock
    def display_external_forces(self,
                                visibility: Union[Sequence[bool], bool]
                                ) -> None:
        """Display external forces applied on the joints the robot, as arrows
        of variable length depending of magnitude of the force.

        .. warning::
            It only display the linear component of the force, while ignoring
            the angular part for now.

        .. warning::
            It corresponds to the attribute ``viewer.f_external`. Calling
            `Viewer.display` will NOT update its value automatically.  It is up
            to the user to keep it up-to-date.

        .. warning::
            This method is only supported by Panda3d.

        :param visibility: Whether or not to display the external force applied
                           at each joint selectively. If a boolean is provided,
                           the same visibility will be set for each joint,
                           alternatively, one can provide a boolean list whose
                           ordering is consistent with pinocchio model (i.e.
                           `pinocchio_model.names`).
        """
        # Make sure the current backend is supported by this method
        if not Viewer.backend.startswith('panda3d'):
            raise NotImplementedError(
                "This method is only supported by Panda3d.")

        # Convert boolean visiblity to mask if necessary
        if isinstance(visibility, bool):
            visibility = [visibility] * (self._client.model.njoints - 1)

        # Check that the length of the mask is consistent with the model
        assert len(visibility) == self._client.model.njoints - 1, (
            "The length of the visibility mask must be equal to the number of "
            "joints of the model, 'universe' excluded.")

        # Update visibility
        for i in range(self._client.model.njoints - 1):
            name = f"ForceExternal_{self._client.model.names[i + 1]}"
            self._gui.show_node(self._markers_group, name, visibility[i])
            self._markers_visibility[name] = visibility[i]
        self._display_f_external = list(visibility)

        # Must refresh the scene
        if any(visibility):
            self.refresh()

    @__must_be_open
    @__with_lock
    def remove_marker(self, name: str) -> None:
        """Remove a marker, based on its name.

        :param identifier: Name of the marker to remove.
        """
        if name not in self.markers.keys():
            raise ValueError(f"Marker's name '{name}' does not exists.")
        self.markers.pop(name)
        self._gui.remove_node(self._markers_group, name)

    @__must_be_open
    @__with_lock
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
        # Extract pinocchio model and data pairs to update
        model_list, data_list, model_type_list = [], [], []
        if self._client.display_collisions or force_update_collision:
            model_list.append(self._client.collision_model)
            data_list.append(self._client.collision_data)
            model_type_list.append(pin.GeometryType.COLLISION)
        if self._client.display_visuals or force_update_visual:
            model_list.append(self._client.visual_model)
            data_list.append(self._client.visual_data)
            model_type_list.append(pin.GeometryType.VISUAL)

        # Update geometries placements
        for model, data, in zip(model_list, data_list):
            pin.updateGeometryPlacements(
                self._client.model, self._client.data, model, data)

        # Render new geometries placements
        if Viewer.backend.startswith('panda3d'):
            for geom_model, geom_data, model_type in zip(
                    model_list, data_list, model_type_list):
                pose_dict = {}
                for i, geom in enumerate(geom_model.geometryObjects):
                    oMg = geom_data.oMg[i]
                    x, y, z, qx, qy, qz, qw = SE3ToXYZQUAT(oMg)
                    group, nodeName = self._client.getViewerNodeName(
                        geom, model_type)
                    pose_dict[nodeName] = ((x, y, z), (qw, qx, qy, qz))
                self._gui.move_nodes(group, pose_dict)
        else:
            for geom_model, geom_data, model_type in zip(
                    model_list, data_list, model_type_list):
                for i, geom in enumerate(geom_model.geometryObjects):
                    oMg = geom_data.oMg[i]
                    S = np.diag((*geom.meshScale, 1.0))
                    T = oMg.homogeneous.dot(S)
                    nodeName = self._client.getViewerNodeName(
                        geom, model_type)
                    self._gui[nodeName].set_transform(T)

        # Update the camera placement if necessary
        if Viewer._camera_travelling is not None:
            if Viewer._camera_travelling['viewer'] is self:
                if Viewer._camera_travelling['pose'] is not None:
                    self.set_camera_transform(
                        *Viewer._camera_travelling['pose'],
                        relative=Viewer._camera_travelling['frame'])
                else:
                    frame = Viewer._camera_travelling['frame']
                    self.set_camera_lookat(np.zeros(3), frame)

        if Viewer._camera_motion is not None:
            self.set_camera_transform(*Viewer._camera_xyzrpy)

        # Update pose, color and scale of the markers, if any
        if Viewer.backend.startswith('panda3d'):
            pose_dict, material_dict, scale_dict = {}, {}, {}
            for marker_name, marker_data in self.markers.items():
                if not self._markers_visibility[marker_name]:
                    continue
                marker_data = {key: value() if callable(value) else value
                               for key, value in marker_data.items()}
                (x, y, z), orientation = marker_data["pose"]
                if orientation.ndim > 1:
                    qx, qy, qz, qw = pin.Quaternion(orientation).coeffs()
                else:
                    qx, qy, qz, qw = orientation
                pose_dict[marker_name] = ((x, y, z), (qw, qx, qy, qz))
                color = marker_data["color"]
                if color is not None:
                    r, g, b, a = marker_data["color"]
                    material_dict[marker_name] = (2.0 * r, 2.0 * g, 2.0 * b, a)
                scale_dict[marker_name] = marker_data["scale"]
            self._gui.move_nodes(self._markers_group, pose_dict)
            self._gui.set_materials(self._markers_group, material_dict)
            self._gui.set_scales(self._markers_group, scale_dict)

        # Wait for the backend viewer to finish rendering if requested
        if wait:
            Viewer.wait(require_client=False)

    @__must_be_open
    def display(self,
                q: np.ndarray,
                v: Optional[np.ndarray] = None,
                xyz_offset: Optional[np.ndarray] = None,
                update_hook: Optional[Callable[[], None]] = None,
                wait: bool = False) -> None:
        """Update the configuration of the robot.

        .. warning::
            It will alter original robot data if viewer attribute
            `use_theoretical_model` is false.

        :param q: Configuration of the robot.
        :param v: Velocity of the robot. Used only to update velocity
                  dependent markers such as DCM. `None` if undefined.
                  Optional: `None` by default.
        :param xyz_offset: Freeflyer position offset. Note that it does not
                           check for the robot actually have a freeflyer.
        :param update_hook: Callable that will be called right after updating
                            kinematics data. `None` to disable.
                            Optional: None by default.
        :param wait: Whether or not to wait for rendering to finish.
        """
        assert self._client.model.nq == q.shape[0], (
            "The configuration vector does not have the right size.")

        # Make sure the state is valid
        if np.isnan(q).any() or (v is not None and np.isnan(v).any()):
            raise ValueError("The input state ('q','v') contains 'nan'.")

        # Apply offset on the freeflyer, if requested.
        # Note that it is NOT checking that the robot has a freeflyer.
        if xyz_offset is not None:
            q = q.copy()  # Make a copy to avoid altering the original data
            q[:3] += xyz_offset

        # Update pinocchio data
        pin.framesForwardKinematics(self._client.model, self._client.data, q)
        if v is None:
            pin.centerOfMass(self._client.model, self._client.data, q, False)
        else:
            pin.centerOfMass(
                self._client.model, self._client.data, q, v, False)

        # Call custom update hook
        if update_hook is not None:
            update_hook()

        # Refresh the viewer
        self.refresh(wait)

    @__must_be_open
    def replay(self,
               evolution_robot: Sequence[State],
               time_interval: Optional[Union[
                   np.ndarray, Tuple[float, float]]] = (0.0, np.inf),
               speed_ratio: float = 1.0,
               xyz_offset: Optional[np.ndarray] = None,
               update_hook: Optional[Callable[
                   [float, np.ndarray, np.ndarray], None]] = None,
               enable_clock: bool = False,
               wait: bool = False) -> None:
        """Replay a complete robot trajectory at a given real-time ratio.

        .. note::
            Specifying 'udpate_hook' is necessary to be able to display sensor
            information such as contact forces. It will be automatically
            disable otherwise.

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
        :param update_hook: Callable that will be called periodically between
                            every state update. `None` to disable, otherwise it
                            must have the following signature:
                                f(t:float, q: ndarray, v: ndarray) -> None
                            Optional: No update hook by default.
        :param wait: Whether or not to wait for rendering to finish.
        """
        # Disable display of sensor data if no update hook is provided
        disable_display_contact_forces = False
        if update_hook is None and self._display_contact_forces:
            disable_display_contact_forces = True
            self.display_contact_forces(False)

        # Disable display of DCM if no velocity data provided
        disable_display_dcm = False
        has_velocities = evolution_robot[0].v is not None
        if not has_velocities and self._display_dcm:
            disable_display_dcm = True
            self.display_capture_point(False)

        # Check if force data is available
        has_forces = evolution_robot[0].f_ext is not None

        # Replay the whole trajectory at constant speed ratio
        v = None
        update_hook_t = None
        times = [s.t for s in evolution_robot]
        t_simu = time_interval[0]
        i = bisect_right(times, t_simu)
        time_init = time.time()
        time_prev = time_init
        while i < len(evolution_robot):
            try:
                # Update clock if enabled
                if enable_clock:
                    Viewer.set_clock(t_simu)

                # Compute interpolated data at current time
                s_next = evolution_robot[min(i, len(times) - 1)]
                s = evolution_robot[max(i - 1, 0)]
                ratio = (t_simu - s.t) / (s_next.t - s.t)
                q = pin.interpolate(self._client.model, s.q, s_next.q, ratio)
                if has_velocities:
                    v = s.v + ratio * (s_next.v - s.v)
                if has_forces:
                    for i, (f_ext, f_ext_next) in enumerate(zip(
                            s.f_ext, s_next.f_ext)):
                        self.f_external[i].vector[:] = \
                            f_ext + ratio * (f_ext_next - f_ext)

                # Update camera motion
                if Viewer._camera_motion is not None:
                    Viewer._camera_xyzrpy = Viewer._camera_motion(t_simu)

                # Update display
                if update_hook is not None:
                    update_hook_t = partial(update_hook, t_simu, q, v)
                self.display(q, v, xyz_offset, update_hook_t, wait)

                # Sleep for a while if computing faster than display framerate
                sleep(1.0 / REPLAY_FRAMERATE - (time.time() - time_prev))

                # Update time in simulation, taking into account speed ratio
                time_prev = time.time()
                time_elapsed = time_prev - time_init
                t_simu = time_interval[0] + speed_ratio * time_elapsed

                # Compute corresponding right index from interpolation
                i = bisect_right(times, t_simu)

                # Waiting for the first timestep is enough
                wait = False

                # Stop the simulation if final time is reached
                if t_simu > time_interval[1]:
                    break
            except (ViewerClosedError, zmq.error.Again, zmq.error.ZMQError):
                # Make sure the viewer is properly closed if exception is
                # raised during replay.
                Viewer.close()
                return

        # Restore Viewer's state if it has been altered
        if Viewer.is_alive():
            # Disable clock after replay if enabled
            if enable_clock:
                Viewer.set_clock()

            # Restore display if necessary
            if disable_display_contact_forces:
                self.display_contact_forces(True)
            if disable_display_dcm:
                self.display_capture_point(True)
