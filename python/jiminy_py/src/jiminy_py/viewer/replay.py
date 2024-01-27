# mypy: disable-error-code="attr-defined, name-defined"
""" TODO: Write documentation.
"""
import os
import time
import ctypes
import logging
import pathlib
import asyncio
import tempfile
import argparse
from base64 import b64encode
from bisect import bisect_right
from collections import deque
from types import TracebackType
from functools import wraps, partial
from threading import get_ident, Thread, Lock
from itertools import cycle, islice
from typing import (
    cast, List, Dict, Deque, Any, Optional, Union, Sequence, Tuple, Callable,
    Type)

import av
import numpy as np
from tqdm import tqdm

import pinocchio as pin

from .. import core as jiminy
from ..dynamics import TrajectoryDataType
from ..log import (read_log,
                   build_robot_from_log,
                   extract_trajectory_from_log,
                   update_sensors_data_from_log)
from .viewer import (COLORS,
                     Tuple3FType,
                     Tuple4FType,
                     CameraPoseType,
                     CameraMotionType,
                     interp1d,
                     get_default_backend,
                     is_display_available,
                     Viewer)
from .meshcat.utilities import interactive_mode


VIDEO_FRAMERATE = 30
VIDEO_QUALITY = 0.3  # [Mbytes/s]

LOGGER = logging.getLogger(__name__)

ColorType = Union[str, Tuple4FType]


class QRLock:
    """Fair (FIFO) reentrant lock.

    It is similar to the built-in `threading.RLock` object that is also
    reentrant but unfair.

    .. seealso::
        For reference: https://stackoverflow.com/a/19695878/4820605
    """
    def __init__(self) -> None:
        self._lock = Lock()
        self._waiters: Deque = deque()
        self._owner: Optional[int] = None
        self._count = 0

    def acquire(self) -> None:
        """Acquire a lock in a blocking way.

        .. note::
            If this thread already owns the lock, increment the recursion level
            by one, and return immediately. Otherwise, if another thread owns
            the lock, block until the lock is unlocked. Once the lock is
            unlocked (not owned by any thread), then grab ownership, set the
            recursion level to one, and return. If more than one thread is
            blocked waiting until the lock is unlocked, only one at a time will
            be able to grab ownership of the lock.

        .. warning::
            Unlike built-in `threading.RLock`, if multiple threads are waiting
            for acquiring the lock, which one will get it next is not system
            dependent. The priority is given to the first thread in queue that
            waited for it.
        """
        # pylint: disable=consider-using-with
        thread_id = get_ident()
        if self._owner == thread_id:
            self._count += 1
            return
        self._lock.acquire()
        if self._count:
            new_lock = Lock()
            new_lock.acquire()
            self._waiters.append(new_lock)
            self._lock.release()
            new_lock.acquire()
            self._lock.acquire()
        self._owner = thread_id
        self._count += 1
        self._lock.release()

    def release(self) -> None:
        """Release a lock, decrementing the recursion level.

        .. note::
            If after the decrement it is zero, reset the lock to unlocked (not
            owned by any thread), and if any other threads are blocked waiting
            for the lock to become unlocked, allow exactly one of them to
            proceed. If after the decrement the recursion level is still
            nonzero, the lock remains locked and owned by the calling thread.

        .. warning::
            Only call this method when the calling thread owns the lock. A
            RuntimeError is raised if this method is called when the lock is
            unlocked.
        """
        with self._lock:
            if self._owner != get_ident():
                raise RuntimeError("cannot release un-acquired lock")
            self._count = count = self._count - 1
            if not count:
                self._owner = None
                if self._waiters:
                    self._waiters.popleft().release()

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.release()


viewer_lock = QRLock()


def _with_lock(fun: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fun)
    def fun_safe(*args: Any, **kwargs: Any) -> Any:
        with viewer_lock:
            return fun(*args, **kwargs)
    return fun_safe


@_with_lock
def play_trajectories(
        trajs_data: Union[  # pylint: disable=unused-argument
            TrajectoryDataType, Sequence[TrajectoryDataType]],
        update_hooks: Optional[Union[
            Callable[[float, np.ndarray, np.ndarray], None],
            Sequence[Optional[Callable[[float, np.ndarray, np.ndarray], None]]]
            ]] = None,
        time_interval: Union[np.ndarray, Tuple[float, float]] = (0.0, np.inf),
        speed_ratio: float = 1.0,
        xyz_offsets: Optional[
            Union[Tuple3FType, Sequence[Optional[Tuple3FType]]]] = None,
        robots_colors: Optional[
            Union[ColorType, Sequence[Optional[ColorType]]]] = None,
        camera_pose: Optional[CameraPoseType] = None,
        enable_travelling: bool = False,
        camera_motion: Optional[CameraMotionType] = None,
        watermark_fullpath: Optional[str] = None,
        legend: Optional[Union[str, Sequence[str]]] = None,
        enable_clock: bool = False,
        display_com: Optional[bool] = None,
        display_dcm: Optional[bool] = None,
        display_contacts: Optional[bool] = None,
        display_f_external: Optional[Union[Sequence[bool], bool]] = None,
        scene_name: str = 'world',
        record_video_path: Optional[str] = None,
        record_video_size: Optional[Tuple[int, int]] = None,
        start_paused: bool = False,
        backend: Optional[str] = None,
        delete_robot_on_close: Optional[bool] = None,
        remove_widgets_overlay: bool = True,
        close_backend: bool = False,
        viewers: Optional[Union[Viewer, Sequence[Optional[Viewer]]]] = None,
        verbose: bool = True,
        **kwargs: Any) -> Sequence[Viewer]:
    """Replay one or several robot trajectories in a viewer.

    The ratio between the replay and the simulation time is kept constant to
    the desired ratio. One can choose between several backend ('panda3d' or
    'meshcat').

    .. note::
        Replay speed is independent of the platform (windows, linux...) and
        available CPU power.

    :param trajs_data: List of `TrajectoryDataType` dicts.
    :param update_hooks: Callables associated with each robot that can be used
                         to update non-kinematic robot data, for instance to
                         emulate sensors data from log using the hook provided
                         by `update_sensors_data_from_log` method. `None` to
                         disable, otherwise it must have the signature:

                         .. code-block:: python

                             f(t:float, q: ndarray, v: ndarray) -> None

                         Optional: None by default.
    :param time_interval: Replay only timesteps in this interval of time.
                          It does not have to be finite.
                          Optional: [0, inf] by default.
    :param speed_ratio: Speed ratio of the simulation.
                        Optional: 1.0 by default.
    :param xyz_offsets: List of constant position of the root joint for each
                        robot in world frame. None to disable.
                        Optional: None by default.
    :param robots_colors: List of RGBA codes or named colors defining the color
                          for each robot. It will be applied to every link.
                          None to disable.
                          Optional: Original color if single robot, default
                          color cycle otherwise.
    :param camera_pose: Tuple position [X, Y, Z], rotation [Roll, Pitch, Yaw],
                        frame name/index specifying the initial pose of the
                        camera or the relative pose wrt the tracked frame
                        depending on whether travelling is enabled. `None` to
                        disable.
                        Optional: None by default.
    :param enable_travelling: Whether the camera tracks the robot associated
                              with the first trajectory specified in
                              `trajs_data`. `None` to disable.
                              Optional: Disabled by default.
    :param camera_motion: Camera breakpoint poses over time, as a list of
                          `CameraMotionBreakpointType` dict. None to disable.
                          Optional: None by default.
    :param watermark_fullpath: Add watermark to the viewer. It is not
                               persistent but disabled after replay. This
                               option is only supported by Meshcat backend.
                               None to disable.
                               Optional: No watermark by default.
    :param legend: List of labels defining the legend for each robot. It is not
                   persistent but disabled after replay. None to disable.
                   Optional: No legend by default for a single robot without
                   color, the name of each robot otherwise.
    :param enable_clock: Add clock on bottom right corner of the viewer.
                         Only available with 'panda3d' rendering backend.
                         Optional: Disabled by default.
    :param display_com: Whether to display the center of mass. `None` to keep
                        current viewers' settings, if any.
                        Optional: Enabled by default iif `viewers` is `None`,
                        and backend is 'panda3d'.
    :param display_dcm: Whether to display the capture point (also called DCM).
                        `None to keep current viewers' settings.
                        Optional: Enabled by default iif `viewers` is `None`,
                        and backend is 'panda3d'.
    :param display_contacts: Whether to display the contact forces. Note that
                             the user is responsible for updating sensors data
                             via `update_hooks`. `None` to keep current
                             viewers' settings.
                             Optional: Enabled by default iif `update_hooks` is
                             specified, `viewers` is `None`, and backend is
                             'panda3d'.
    :param display_f_external: Whether to display the external external forces
                               applied at the joints on the robot. If a boolean
                               is provided, the same visibility will be set for
                               each joint, alternatively one can provide a
                               boolean list whose ordering is consistent with
                               `pinocchio_model.names`. Note that the user is
                               responsible for updating the force buffer
                               `viewer.f_external` via `update_hooks`. `None`
                               to keep current viewers' settings.
                               Optional: `None` by default.
    :param scene_name: Name of viewer's scene in which to display the robot.
                       Optional: Common default name if omitted.
    :param record_video_path: Fullpath location where to save generated video.
                              It must be specified to enable video recording.
                              Meshcat only support 'webm' format, while the
                              other renderer only supports 'mp4' format encoded
                              with web-compatible 'h264' codec.
                              Optional: None by default.
    :param record_video_size: The width and height of the video recording.
                              Optional: (800, 800) if non-interactive and
                              (800, 400) otherwise by default.
    :param start_paused: Start the simulation is pause, waiting for keyboard
                         input before starting to play the trajectories.
                         Only available if `record_video_path` is None.
                         Optional: False by default.
    :param backend: Backend, one of 'meshcat', 'panda3d' and'panda3d-sync. If
                    `None`, the most appropriate backend will be selected
                    automatically based on hardware and python environment.
                    Optional: `None` by default.
    :param delete_robot_on_close: Whether to delete the robot from the viewer
                                  when closing it.
                                  Optional: True by default for Panda3d backend
                                  in non-interactive mode, False otherwise.
    :param remove_widgets_overlay: Remove overlay (legend, watermark, clock,
                                   ...) automatically before returning.
                                   Optional: Enabled by default.
    :param close_backend: Whether to close backend automatically before
                          returning.
                          Optional: Disabled by default.
    :param viewers: List of already instantiated viewers, associated one by one
                    in order to each trajectory data. None to disable.
                    Optional: None by default.
    :param verbose: Add information to keep track of the process.
                    Optional: True by default.
    :param kwargs: Unused keyword arguments to allow chaining rendering
                   methods with ease.

    :returns: List of viewers used to play the trajectories.
    """
    # Make sure sequence arguments are list or tuple
    if isinstance(trajs_data, dict):
        trajs_data = [trajs_data]
    if update_hooks is None:
        update_hooks = [None] * len(trajs_data)
    if callable(update_hooks):
        update_hooks = [update_hooks]
    if isinstance(viewers, Viewer):
        viewers = [viewers]
    elif not viewers:
        viewers = None

    # Make sure the viewers are still running if specified
    if not Viewer.is_alive():
        viewers = None
    if viewers is not None:
        viewers = list(viewers)
        for i, viewer in enumerate(viewers):
            if (viewer is not None and
                    not viewer.is_open()):  # type: ignore[misc]
                viewers[i] = None
                break
        if all(viewer is None for viewer in viewers):
            viewers = None

    # Pick the default backend if unspecified and none is already running. Note
    # that repeatedly switching between "panda3d-sync" and "panda3d" backends
    # is causing segfaults. See https://github.com/panda3d/panda3d/issues/1372.
    if backend is None:
        if Viewer.is_alive():
            backend = Viewer.backend
        # elif record_video_path is not None:
        #     backend = "panda3d-sync"
        else:
            backend = get_default_backend()
    assert backend is not None

    # Always close viewer if gui is open if necessary for efficiency
    if (backend.startswith("panda3d") and record_video_path is not None and
            Viewer.has_gui()):
        Viewer.close()

    # Create a temporary video if the backend is 'panda3d-sync', no
    # 'record_video_path' is provided, and running in interactive mode
    # with HTML rendering support. Then load it in running cell.
    record_video_html_embedded = (
        record_video_path is None and
        backend == "panda3d-sync" and interactive_mode() >= 2)
    if record_video_html_embedded:
        fd, record_video_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)

    # Make sure it is possible to replay something
    if backend == "panda3d-sync" and record_video_path is None:
        raise RuntimeError(
            "Impossible to replay simulation using 'panda3d-sync' backend. "
            "Please set 'record_video_path' to save it as a file.")

    # Set default video recording size
    if record_video_size is None:
        record_video_size = (800, 400 if record_video_html_embedded else 800)

    # Delete robot by default only if not in interactive viewer
    if delete_robot_on_close is None:
        delete_robot_on_close = (
            backend.startswith('panda3d') and interactive_mode() < 2)

    # Handling of default options if no viewer is available
    if viewers is None:
        # Handling of default display of CoM, DCM and contact forces
        if backend.startswith('panda3d'):
            if display_com is None:
                display_com = True
            if display_dcm is None:
                display_dcm = True
            if display_contacts is None:
                display_contacts = all(fun is not None for fun in update_hooks)

    # Make sure it is possible to display contacts if requested
    if display_contacts:
        for traj in trajs_data:
            robot = traj['robot']
            assert robot is not None
            if robot.is_locked:
                LOGGER.debug(
                    "`display_contacts` is not available if robot is locked. "
                    "Please stop any running simulation before replay.")
                display_contacts = False
                break

    # Sanitize user-specified robot offsets
    if xyz_offsets is None:
        xyz_offsets = len(trajs_data) * (None,)
    elif len(xyz_offsets) != len(trajs_data):
        assert isinstance(xyz_offsets[0], float)
        xyz_offsets = np.tile(
            xyz_offsets, (len(trajs_data), 1))  # type: ignore[arg-type]
    assert xyz_offsets is not None

    # Sanitize user-specified robot colors
    if robots_colors is None:
        if len(trajs_data) == 1:
            robots_colors = (None,)
        else:
            robots_colors = list(islice(
                cycle(COLORS.values()), len(trajs_data)))
    elif isinstance(robots_colors, str) or isinstance(robots_colors[0], float):
        robots_colors = [robots_colors]  # type: ignore[list-item]
    assert len(robots_colors) == len(trajs_data)

    # Sanitize user-specified legend
    if legend is not None and isinstance(legend, str):
        legend = [legend]

    # Make sure the viewers instances are consistent with the trajectories
    if viewers is None:
        viewers = [None for _ in trajs_data]
    assert len(viewers) == len(trajs_data)

    # Instantiate or refresh viewers if necessary
    for i, (viewer, traj, color) in enumerate(zip(
            viewers, trajs_data, robots_colors)):
        # Extract robot from trajectory
        robot = traj['robot']
        assert robot is not None

        # Create new viewer instance if necessary, and load the robot in it
        if viewer is None:
            uniq_id = next(tempfile._get_candidate_names())
            robot_name = f"{uniq_id}_robot_{i}"
            use_theoretical_model = traj['use_theoretical_model']
            viewer = Viewer(
                robot,
                use_theoretical_model=use_theoretical_model,
                robot_color=color,
                robot_name=robot_name,
                backend=backend,
                scene_name=scene_name,
                delete_robot_on_close=delete_robot_on_close,
                open_gui_if_parent=record_video_path is None)
            viewers[i] = viewer

        # Reset the configuration of the robot
        if traj['use_theoretical_model']:
            model = robot.pinocchio_model_th
        else:
            model = robot.pinocchio_model
        viewer.display(pin.neutral(model))

        # Reset robot model in viewer if requested color has changed.
        # `_setup` is called instead of `set_color` because the latter is not
        # supported by meshcat.
        if color is not None and color != viewer.robot_color:
            viewer._setup(robot, color)

    # Assert(s) for type checker
    assert all(viewers)
    viewers = cast(List[Viewer], viewers)

    # Add default legend with robots names if replaying multiple trajectories
    if all(color is not None for color in robots_colors) and legend is None:
        legend = [viewer.robot_name for viewer in viewers]

    # Use first viewers as main viewer to call static methods conveniently
    viewer = viewers[0]
    assert Viewer._backend_obj is not None

    # Make sure clock is only enabled for panda3d backend
    if enable_clock and not backend.startswith('panda3d'):
        LOGGER.warning(
            "`enable_clock` is only available with 'panda3d' backend.")
        enable_clock = False

    # Early return if nothing to replay
    if all(not traj['evolution_robot'] for traj in trajs_data):
        LOGGER.debug("Nothing to replay.")
        return viewers

    # Enable camera motion if requested
    if camera_motion is not None:
        Viewer.register_camera_motion(camera_motion)

    # Handle meshcat-specific options
    if legend is not None:
        try:
            Viewer.set_legend(legend)
        except ImportError:
            LOGGER.warning(
                "Impossible to add legend. Please install 'jiminy_py[plot]'.")
            legend = None

    # Add watermark if requested
    if watermark_fullpath is not None:
        Viewer.set_watermark(watermark_fullpath)

    # Make sure the time interval is valid
    if time_interval[1] < time_interval[0]:
        raise ValueError("Time interval must be non-empty and positive.")

    # Initialize robot configuration is viewer before any further processing
    for viewer_i, traj, offset in zip(viewers, trajs_data, xyz_offsets):
        data = traj['evolution_robot']
        if data:
            i = bisect_right(
                [s.t for s in data], time_interval[0], hi=len(data)-1)
            for f_ext in viewer_i.f_external:
                f_ext.vector[:] = 0.0
            viewer_i.display(data[i].q, data[i].v, offset)
        if backend.startswith('panda3d'):
            if display_com is not None:
                viewer_i.display_center_of_mass(display_com)
            if display_dcm is not None:
                viewer_i.display_capture_point(display_dcm)
            if display_contacts is not None:
                viewer_i.display_contact_forces(display_contacts)
            if display_f_external is not None:
                viewer_i.display_external_forces(display_f_external)

    # Set camera pose or activate camera travelling if requested
    if enable_travelling:
        position, rotation, relative = None, None, None
        if camera_pose is not None:
            position, rotation, relative = camera_pose
        if relative is None:
            # Track the first actual frame by default (0: world, 1: root_joint)
            robot = trajs_data[0]['robot']
            assert robot is not None
            if not robot.has_freeflyer:
                raise ValueError(
                    "Enabling travelling requires `camera_pose` to specify at "
                    "least the relative frame to track if the first robot has "
                    "no freeflyer.")
            relative = 2
        viewer.attach_camera(relative, position, rotation)
    elif camera_pose is not None:
        viewer.set_camera_transform(*camera_pose)

    # Wait for the meshes to finish loading if video recording is disable
    if record_video_path is None:
        if backend == 'meshcat':
            if verbose and interactive_mode() < 2:
                print("Waiting for meshcat client in browser to connect: "
                      f"{Viewer._backend_obj.gui.url()}")
            Viewer.wait(require_client=True)
            if verbose and interactive_mode() < 2:
                print("Browser connected! Replaying simulation...")

    # Replay the trajectory
    if record_video_path is not None:
        # Extract and resample trajectory data at fixed framerate
        time_max = time_interval[0]
        for traj in trajs_data:
            if len(traj['evolution_robot']):
                time_max = max([time_max, traj['evolution_robot'][-1].t])
        time_max = min(time_max, time_interval[1])

        time_global = np.arange(
            time_interval[0], time_max, speed_ratio / VIDEO_FRAMERATE)
        position_evolutions: List[Optional[np.ndarray]] = []
        velocity_evolutions: List[Optional[
            Union[np.ndarray, Sequence[None]]]] = []
        force_evolutions: List[Optional[
            Union[List[List[np.ndarray]], Sequence[None]]]] = []
        for traj in trajs_data:
            if len(traj['evolution_robot']):
                data_orig = traj['evolution_robot']
                robot = traj['robot']
                assert robot is not None
                if traj['use_theoretical_model']:
                    model = robot.pinocchio_model_th
                else:
                    model = robot.pinocchio_model
                t_orig = np.array([s.t for s in data_orig])
                pos_orig = np.stack([s.q for s in data_orig], axis=0)
                position_evolutions.append(jiminy.interpolate(
                    model, t_orig, pos_orig, time_global))
                if data_orig[0].v is not None:
                    vel_orig = np.stack([
                        s.v  # type: ignore[misc]
                        for s in data_orig], axis=0)
                    velocity_evolutions.append(interp1d(
                        t_orig, vel_orig, time_global))
                else:
                    velocity_evolutions.append((None,) * len(time_global))
                if data_orig[0].f_ext is not None:
                    forces: List[np.ndarray] = []
                    for i in range(len(data_orig[0].f_ext)):
                        f_ext_orig = np.stack([
                            s.f_ext[i]  # type: ignore[index]
                            for s in data_orig], axis=0)
                        forces.append(interp1d(
                            t_orig, f_ext_orig, time_global))
                    force_evolutions.append([
                        [f_ext[i] for f_ext in forces]
                        for i in range(len(time_global))])
                else:
                    force_evolutions.append((None,) * len(time_global))
            else:
                position_evolutions.append(None)
                velocity_evolutions.append(None)
                force_evolutions.append(None)

        # Initialize video recording
        if backend == 'meshcat':
            # Sanitize the recording path to enforce '.webm' extension
            record_video_path = str(
                pathlib.Path(record_video_path).with_suffix('.webm'))

            # Start backend recording thread
            Viewer._backend_obj.start_recording(
                VIDEO_FRAMERATE, *record_video_size)
        else:
            # Sanitize the recording path to enforce '.mp4' extension
            record_video_path = str(
                pathlib.Path(record_video_path).with_suffix('.mp4'))

            # Create ffmpeg video writer
            out = av.open(record_video_path, mode='w')
            out.metadata['title'] = scene_name
            stream = out.add_stream('libx264', rate=VIDEO_FRAMERATE)
            stream.width, stream.height = record_video_size
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = VIDEO_QUALITY * (8 * 1024 ** 2)
            stream.options = {"preset": "veryfast", "tune": "zerolatency"}

            # Create frame storage
            frame = av.VideoFrame(*record_video_size, 'rgb24')

        # Add frames to video sequentially
        for i, t_cur in enumerate(tqdm(
                time_global, desc="Rendering frames",
                disable=(not verbose and not record_video_html_embedded))):
            try:
                # Update 3D view
                for viewer, pos, vel, forces, xyz_offset, update_hook in zip(
                        viewers, position_evolutions, velocity_evolutions,
                        force_evolutions, xyz_offsets, update_hooks):
                    assert viewer is not None
                    if pos is None:
                        continue
                    q, v, f_ext = pos[i], vel[i], forces[i]
                    if f_ext is not None:
                        for f_ref, f_i in zip(viewer.f_external, f_ext):
                            f_ref.vector[:] = f_i
                    if update_hook is not None:
                        update_hook_t = partial(update_hook, t_cur, q, v)
                    else:
                        update_hook_t = None
                    viewer.display(q, v, xyz_offset, update_hook_t)

                # Update clock if enabled
                if enable_clock:
                    Viewer.set_clock(t_cur)

                # Add frame to video
                if backend == 'meshcat':
                    Viewer._backend_obj.add_frame()
                else:
                    # Update frame.
                    # Note that `capture_frame` is by far the main bottleneck
                    # of the whole recording process (~75% on discrete gpu).
                    buffer = Viewer.capture_frame(
                        *record_video_size, raw_data=True)
                    memoryview(frame.planes[0])[:] = buffer

                    # Write frame
                    for packet in stream.encode(frame):
                        out.mux(packet)
            except (KeyboardInterrupt, RuntimeError):
                # RuntimeError would be raised if the backend got closed during
                # recording, which typically happens when terminating Python
                # forcibly during async recording.
                break

        # Finalize video recording
        if backend == 'meshcat':
            # Stop backend recording thread
            assert record_video_path is not None
            Viewer._backend_obj.stop_recording(record_video_path)
        else:
            # Flush and close recording file
            for packet in stream.encode(None):
                out.mux(packet)
            out.close()
    else:
        # Make sure a gui is opened
        Viewer.open_gui()

        # Handle start-in-pause mode
        if start_paused:
            if interactive_mode() < 2:
                input("Press Enter to continue...")
                if not Viewer.is_alive():
                    return viewers
            else:
                LOGGER.warning("Start paused is disabled in interactive mode.")

        # Play trajectories with multithreading
        def replay_thread(viewer: Viewer, *args: Any) -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            viewer.replay(*args)

        threads = []
        for viewer, traj, xyz_offset, update_hook in zip(
                viewers, trajs_data, xyz_offsets, update_hooks):
            threads.append(Thread(
                target=replay_thread,
                args=(viewer,
                      traj['evolution_robot'],
                      time_interval,
                      speed_ratio,
                      xyz_offset,
                      update_hook,
                      enable_clock),
                daemon=True))
        for thread in threads:
            thread.start()
        for thread in threads:
            try:
                thread.join()
            except KeyboardInterrupt:
                assert thread.ident is not None
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(thread.ident), ctypes.py_object(SystemExit))

    if Viewer.is_alive():
        # Disable camera travelling and camera motion if it was enabled
        if enable_travelling:
            Viewer.detach_camera()
        if camera_motion is not None:
            Viewer.remove_camera_motion()

        if remove_widgets_overlay:
            # Disable legend if it was enabled
            if legend is not None:
                Viewer.set_legend()

            # Disable watermark if it was enabled
            if watermark_fullpath is not None:
                Viewer.set_watermark()

            if enable_clock:
                Viewer.set_clock()

    # Close backend if requested
    if close_backend:
        Viewer.close()

    # Show video if temporary
    if record_video_html_embedded:
        # pylint: disable=import-outside-toplevel,no-name-in-module
        from IPython.core.display import HTML, display
        assert record_video_path is not None
        video_base64 = b64encode(open(record_video_path, 'rb').read()).decode()
        os.remove(record_video_path)
        display(HTML(f"""
        <video controls style="
            height: 400px; width: 100%; overflow: hidden;">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        """))

    return viewers


def extract_replay_data_from_log(
        log_data: Dict[str, np.ndarray],
        robot: jiminy.Robot) -> Tuple[
            TrajectoryDataType,
            Optional[Callable[[float, np.ndarray, np.ndarray], None]],
            Dict[str, Any]]:
    """Extract replay data from log data.

    :param robot: Jiminy robot for which to extract log data.
    :param log_data: Data from the log file, in a dictionary.

    :returns: Trajectory data, update hook and extra keyword arguments to
              forward to `play_trajectories` method to display the trajectory.
              By default, it enables display of external forces applied on
              freeflyer if any.
    """
    # For each pair (log, robot), extract a trajectory object for
    # `play_trajectories`
    trajectory = extract_trajectory_from_log(log_data, robot)

    # Display external forces on root joint, if any
    replay_kwargs = {}
    if robot.has_freeflyer:
        if trajectory['use_theoretical_model']:
            njoints = robot.pinocchio_model_th.njoints
        else:
            njoints = robot.pinocchio_model.njoints
        visibility = [True] + [False] * (njoints - 2)
        replay_kwargs["display_f_external"] = visibility

    # Define `update_hook` to emulate sensor update
    if not robot.is_locked:
        update_hook = update_sensors_data_from_log(log_data, robot)
    else:
        if robot.sensors_names:
            LOGGER.warning(
                "At least one of the robot is locked, which means that a "
                "simulation using the robot is still running. It will be "
                "impossible to display sensor data. Call `simulator.stop` to "
                "unlock the robot before replaying logs data.")
        update_hook = None

    return trajectory, update_hook, replay_kwargs


def play_logs_data(robots: Union[Sequence[jiminy.Robot], jiminy.Robot],
                   logs_data: Union[Sequence[Dict[str, np.ndarray]],
                                    Dict[str, np.ndarray]],
                   **kwargs: Any) -> Sequence[Viewer]:
    """Play log data in a viewer.

    This method simply formats the data then calls `play_trajectories`.

    :param robots: Either a single robot, or a list of robot for each log data.
    :param logs_data: Either a single dictionary, or a list of dictionaries of
                      simulation data log.
    :param kwargs: Keyword arguments to forward to `play_trajectories` method.
    """
    # Reformat input arguments as lists
    if isinstance(logs_data, dict):
        logs_data = [logs_data]
    if isinstance(robots, jiminy.Robot):
        robots = [robots]

    # Extract a replay data for `play_trajectories` for each pair (robot, log)
    trajectories, update_hooks, extra_kwargs = [], [], {}
    for robot, log_data in zip(robots, logs_data):
        traj, update_hook, _kwargs = \
            extract_replay_data_from_log(log_data, robot)
        trajectories.append(traj)
        update_hooks.append(update_hook)
        extra_kwargs.update(_kwargs)

    # Do not display external forces by default if replaying several traj
    if len(trajectories) > 1:
        extra_kwargs.pop("display_f_external", None)

    # Finally, play the trajectories
    return play_trajectories(
        trajectories, update_hooks, **{**extra_kwargs, **kwargs})


def play_logs_files(logs_files: Union[str, Sequence[str]],
                    mesh_path_dir: Optional[str] = None,
                    mesh_package_dirs: Sequence[str] = (),
                    **kwargs: Any) -> Sequence[Viewer]:
    """Play the content of a logfile in a viewer.

    This method reconstruct the exact model used during the simulation
    corresponding to the logfile, including random biases or flexible joints.

    :param logs_files: Either a single simulation log files in any format, or
                       a list.
    :param mesh_path_dir: Overwrite the common root of all absolute mesh paths.
                          It which may be necessary to read log generated on a
                          different environment.
    :param mesh_package_dirs: Additional search paths for all relative mesh
                              paths beginning with 'packages://' directive. It
                              may be necessary to specify it to read log
                              generated on a different environment.
    :param kwargs: Keyword arguments to forward to `play_trajectories` method.
    """
    # Reformat as list
    if isinstance(logs_files, str):
        logs_files = [logs_files]

    # Extract log data and build robot for each log file
    robots, logs_data = [], []
    for log_file in logs_files:
        log_data = read_log(log_file)
        robot = build_robot_from_log(
            log_data, mesh_path_dir, mesh_package_dirs)
        logs_data.append(log_data)
        robots.append(robot)

    # Default legend if several log files are provided
    if "legend" not in kwargs and len(logs_files) > 1:
        kwargs["legend"] = [pathlib.Path(log_file).stem.split('_')[-1]
                            for log_file in logs_files]

    # Forward arguments to lower-level method
    return play_logs_data(robots, logs_data, **kwargs)


def async_play_and_record_logs_files(
        logs_files: Union[str, Sequence[str]],
        enable_replay: Optional[bool] = None,
        mesh_path_dir: Optional[str] = None,
        mesh_package_dirs: Sequence[str] = (),
        **kwargs: Any) -> Optional[Thread]:
    """Play and/or replay the content of a logfile in a viewer asynchronously.

    .. note::
        This call can be made blocking at any point in time by calling `join()`
        on the returned `Thread` instance.

    :param logs_files: Simulation log files in any of the supported formats.
    :param enable_replay: Whether to force replay the simulation. It would
                          first replay then record if this option is enabled
                          and `record_video_path` is specified.
                          Optional: True by default if `record_video_path` is
                          not specified and the current backend supports
                          onscreen rendering, False otherwise.
    :param mesh_path_dir: Overwrite the common root of all absolute mesh paths.
                          It which may be necessary to read log generated on a
                          different environment.
    :param mesh_package_dirs: Prepend custom mesh package search path
                              directories to the ones provided by log file.
    :param kwargs: Keyword arguments to forward to `play_logs_files` method.
    """
    # Handling of default argument(s)
    enable_recording = "record_video_path" in kwargs
    if enable_replay is None:
        enable_replay = not enable_recording and (
            (Viewer.backend or get_default_backend()) != "panda3d-sync" or
            interactive_mode() >= 2)

    # Disable replay if not available and video recording is requested
    if enable_replay and not is_display_available():
        LOGGER.warning("No display available. Disabling replay.")
        enable_replay = False

    # Nothing to do. Silently returning early.
    if not enable_recording and not enable_replay:
        return None

    # Define method to pass to threading
    def _locked_play_and_record(lock: QRLock,
                                logs_files: Sequence[str],
                                mesh_path_dir: Optional[str],
                                mesh_package_dirs: Sequence[str],
                                enable_replay: bool,
                                **kwargs: Any) -> None:
        """A lock is used to force waiting for the current evaluation to finish
        before starting a new one, otherwise it will crash because of request
        collisions, and it is undesirable anyway.

        The viewer must be closed after replay if recording is requested,
        otherwise the graphical window will dramatically slowdown rendering.
        """
        close_backend = kwargs.get("close_backend", None)
        record_video_path = kwargs.pop("record_video_path", None)
        with lock:
            if close_backend:
                # Make sure there is no viewer already open at this point.
                # Otherwise adding the legend will fail if some robots have not
                # been deleted from the scene.
                Viewer.close()
            if enable_replay:
                try:
                    play_logs_files(
                        logs_files, mesh_path_dir, mesh_package_dirs, **kwargs)
                except RuntimeError as e:
                    # Replay may fail if current backend does not support it
                    LOGGER.warning(
                        "The current viewer backend '%s' does not support "
                        "replaying simulation: %s", Viewer.backend, e)
            if record_video_path is not None:
                play_logs_files(
                    logs_files, mesh_path_dir, mesh_package_dirs,
                    record_video_path=record_video_path, **kwargs)
            if close_backend:
                # Honor close backend at exit but not between replay and record
                Viewer.close()

    # Start replay and record thread
    thread = Thread(
        target=_locked_play_and_record,
        args=(
            viewer_lock, logs_files, mesh_path_dir, mesh_package_dirs,
            enable_replay),
        kwargs={
            **dict(
                close_backend=(enable_replay and enable_recording),
                verbose=False),
            **kwargs, **dict(
                delete_robot_on_close=True)},
        daemon=True)
    thread.start()

    return thread


def _play_logs_files_entrypoint() -> None:
    """Command-line entrypoint to replay the content of a logfile in a viewer.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=(
        "Replay Jiminy simulation log files in a viewer. Multiple files "
        "can be specified for simultaneous display"))
    parser.add_argument(
        '-p', '--start_paused', action='store_true',
        help="Start in pause, waiting for keyboard input.")
    parser.add_argument(
        '-s', '--speed_ratio', type=float, default=1.0,
        help="Real time to simulation time factor.")
    parser.add_argument(
        '-t', '--travelling', action='store_true',
        help=("Whether to track the root frame of the first robot, "
              "assuming the robot has a freeflyer."))
    parser.add_argument(
        '-b', '--backend', default='panda3d',
        help="Display backend ('panda3d' or 'meshcat').")
    parser.add_argument(
        '-m', '--mesh_path_dir', default=None,
        help="Fullpath location of mesh directory.")
    parser.add_argument(
        '-v', '--record_video_path', default=None,
        help="Fullpath location where to save generated video.")
    options, files = parser.parse_known_args()
    kwargs = vars(options)
    kwargs['logs_files'] = files

    # Map argument name(s)
    kwargs['enable_travelling'] = kwargs.pop('travelling')

    # Replay trajectories
    repeat = True
    viewers = None
    while repeat:
        viewers = play_logs_files(
            viewers=viewers,
            **{**dict(
                remove_widgets_overlay=False),
                **kwargs})
        kwargs["start_paused"] = False
        kwargs.setdefault("camera_pose", None)
        if kwargs["record_video_path"] is None:
            while True:
                reply = input("Do you want to replay again (y/[n])?").lower()
                if not reply or reply in ("y", "n"):
                    break
            repeat = reply == "y"
        else:
            repeat = False

    # Do not exit method as long as a graphical window is open
    while Viewer.has_gui():
        time.sleep(0.5)


__all__ = [
    'extract_replay_data_from_log',
    'play_logs_data',
    'play_logs_files',
    'async_play_and_record_logs_files',
]
