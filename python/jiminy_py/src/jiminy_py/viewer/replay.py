import time
import logging
import pathlib
import asyncio
import tempfile
import argparse
from bisect import bisect_right
from functools import partial
from threading import Thread
from itertools import cycle, islice
from typing import Optional, Union, Sequence, Tuple, Dict, Any, Callable

import av
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d

from .. import core as jiminy
from ..log import (TrajectoryDataType,
                   read_log,
                   build_robot_from_log_constants,
                   extract_trajectory_data_from_log,
                   emulate_sensors_data_from_log)
from .viewer import (COLORS,
                     Tuple3FType,
                     Tuple4FType,
                     CameraPoseType,
                     CameraMotionType,
                     Viewer)
from .meshcat.utilities import interactive_mode


VIDEO_FRAMERATE = 30
VIDEO_SIZE = (800, 800)
VIDEO_QUALITY = 0.3  # [Mbytes/s]


logger = logging.getLogger(__name__)


ColorType = Union[Tuple4FType, str]


def play_trajectories(trajs_data: Union[
                          TrajectoryDataType, Sequence[TrajectoryDataType]],
                      update_hooks: Optional[Union[
                          Callable[[float, np.ndarray, np.ndarray], None],
                          Sequence[Callable[
                              [float, np.ndarray, np.ndarray], None]]]] = None,
                      time_interval: Optional[Union[
                          np.ndarray, Tuple[float, float]]] = (0.0, np.inf),
                      speed_ratio: float = 1.0,
                      xyz_offsets: Optional[Union[
                          Tuple3FType, Sequence[Tuple3FType]]] = None,
                      robots_colors: Optional[Union[
                          ColorType, Sequence[ColorType]]] = None,
                      travelling_frame: Optional[Union[str, int]] = None,
                      camera_xyzrpy: Optional[CameraPoseType] = (None, None),
                      camera_motion: Optional[CameraMotionType] = None,
                      watermark_fullpath: Optional[str] = None,
                      legend: Optional[Union[str, Sequence[str]]] = None,
                      enable_clock: bool = False,
                      display_com: Optional[bool] = None,
                      display_dcm: Optional[bool] = None,
                      display_contacts: Optional[bool] = None,
                      display_f_external: Optional[
                          Union[Sequence[bool], bool]] = None,
                      scene_name: str = 'world',
                      record_video_path: Optional[str] = None,
                      start_paused: bool = False,
                      backend: Optional[str] = None,
                      delete_robot_on_close: Optional[bool] = None,
                      remove_widgets_overlay: bool = True,
                      close_backend: bool = False,
                      viewers: Optional[Sequence[Optional[Viewer]]] = None,
                      verbose: bool = True,
                      **kwargs: Any) -> Sequence[Viewer]:
    """Replay one or several robot trajectories in a viewer.

    The ratio between the replay and the simulation time is kept constant to
    the desired ratio. One can choose between several backend (gepetto-gui or
    meshcat).

    .. note::
        Replay speed is independent of the platform (windows, linux...) and
        available CPU power.

    :param trajs_data: List of `TrajectoryDataType` dicts.
    :param update_hooks: Callables associated with each robot that can be used
                         to update non-kinematic robot data, for instance to
                         emulate sensors data from log using the hook provided
                         by `emulate_sensors_data_from_log` method. `None` to
                         disable, otherwise it must have the signature:
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
    :param travelling_frame: Name or index of the frame to track. The camera
                             will automatically follow the frame of the robot
                             associated with the first `trajs_data`.`None` to
                             disable.
                             Optional: Disabled by default.
    :param camera_xyzrpy: Tuple position [X, Y, Z], rotation [Roll, Pitch, Yaw]
                          corresponding to the absolute pose of the camera
                          during replay, if travelling is disable, or the
                          relative pose wrt the tracked frame otherwise. None
                          to disable.
                          Optional: None by default.
    :param camera_motion: Camera breakpoint poses over time, as a list of
                          `CameraMotionBreakpointType` dict. None to disable.
                          Optional: None by default.
    :param watermark_fullpath: Add watermark to the viewer. It is not
                               persistent but disabled after replay. This
                               option is only supported by meshcat backend.
                               None to disable.
                               Optional: No watermark by default.
    :param legend: List of text defining the legend for each robot. It is not
                   persistent but disabled after replay. This option is only
                   supported by meshcat backend. None to disable.
                   Optional: No legend if no color by default, the robots names
                   otherwise.
    :param enable_clock: Add clock on bottom right corner of the viewer.
                         Only available with 'panda3d' rendering backend.
                         Optional: Disabled by default.
    :param display_com: Whether or not to display the center of mass. `None`
                        to keep current viewers' settings, if any.
                        Optional: Enabled by default iif `viewers` is `None`,
                        and backend is 'panda3d'.
    :param display_dcm: Whether or not to display the capture point (also
                        called DCM). `None to keep current viewers' settings.
                        Optional: Enabled by default iif `viewers` is `None`,
                        and backend is 'panda3d'.
    :param display_contacts: Whether or not to display the contact forces.
                             Note that the user is responsible for updating
                             sensors data via `update_hooks`. `None` to keep
                             current viewers' settings.
                             Optional: Enabled by default iif `update_hooks` is
                             specified, `viewers` is `None`, and backend is
                             'panda3d'.
    :param display_f_external: Whether or not to display the external external
                               forces applied at the joints on the robot. If a
                               boolean is provided, the same visibility will be
                               set for each joint, alternatively one can
                               provide a boolean list whose ordering is
                               consistent with `pinocchio_model.names`. Note
                               that the user is responsible for updating the
                               force buffer `viewer.f_external` via
                               `update_hooks`. `None` to keep current viewers'
                               settings.
                               Optional: `None` by default.
    :param scene_name: Name of viewer's scene in which to display the robot.
                       Optional: Common default name if omitted.
    :param record_video_path: Fullpath location where to save generated video.
                              It must be specified to enable video recording.
                              Meshcat only support 'webm' format, while the
                              other renderer only supports 'mp4' format encoded
                              with web-compatible 'h264' codec.
                              Optional: None by default.
    :param start_paused: Start the simulation is pause, waiting for keyboard
                         input before starting to play the trajectories.
                         Only available if `record_video_path` is None.
                         Optional: False by default.
    :param backend: Backend, one of 'panda3d', 'meshcat', or 'gepetto-gui'. If
                    `None`, the most appropriate backend will be selected
                    automatically, based on hardware and python environment.
                    Optional: `None` by default.
    :param delete_robot_on_close: Whether or not to delete the robot from the
                                  viewer when closing it.
                                  Optional: True by default.
    :param remove_widgets_overlay: Remove overlay (legend, watermark, clock,
                                   ...) automatically before returning.
                                   Optional: Enabled by default.
    :param close_backend: Whether or not to close backend automatically before
                          returning.
                          Optional: Disabled by default.
    :param viewers: List of already instantiated viewers, associated one by one
                    in order to each trajectory data. None to disable.
                    Optional: None by default.
    :param verbose: Add information to keep track of the process.
                    Optional: True by default.
    :param kwargs: Unused keyword arguments to allow chaining renderining
                   methods with ease.

    :returns: List of viewers used to play the trajectories.
    """
    # Make sure sequence arguments are list or tuple
    if not isinstance(trajs_data, (list, tuple)):
        trajs_data = [trajs_data]
    if update_hooks is None:
        update_hooks = [None] * len(trajs_data)
    if not isinstance(update_hooks, (list, tuple)):
        update_hooks = [update_hooks]
    if viewers is None:
        viewers = [None] * len(trajs_data)
    if not isinstance(viewers, (list, tuple)):
        viewers = [viewers]
    if len(viewers) == 0:
        viewers = None

    # Make sure the viewers are still running if specified
    if not Viewer.is_open():
        viewers = None
    if viewers is not None:
        for i, viewer in enumerate(viewers):
            if viewer is not None and not viewer.is_open():
                viewers[i] = None
                break

    # Handling of default options if no viewer is available
    if viewers is None and Viewer.backend.startswith('panda3d'):
        # Delete robot by default only if not in notebook
        if delete_robot_on_close is None:
            delete_robot_on_close = not interactive_mode()

        # Handling of default display of CoM, DCM and contact forces
        if display_com is None:
            display_com = True
        if display_dcm is None:
            display_dcm = True
        if display_contacts is None:
            display_contacts = all(func is not None for func in update_hooks)

    # Make sure it is possible to display contacts if requested
    if display_contacts:
        if any(traj['robot'].is_locked for traj in trajs_data):
            logger.debug(
                "`display_contacts` is not available if robot is locked. "
                "Please stop any running simulation before replay.")
            display_contacts = False

    # Sanitize user-specified robot offsets
    if xyz_offsets is None:
        xyz_offsets = len(trajs_data) * [None]
    elif len(xyz_offsets) != len(trajs_data):
        xyz_offsets = np.tile(xyz_offsets, (len(trajs_data), 1))

    # Sanitize user-specified robot colors
    if robots_colors is None:
        if len(trajs_data) == 1:
            robots_colors = [None]
        else:
            robots_colors = list(islice(
                cycle(COLORS.values()), len(trajs_data)))
    elif not isinstance(robots_colors, (list, tuple)) or \
            isinstance(robots_colors[0], float):
        robots_colors = [robots_colors]
    assert len(robots_colors) == len(trajs_data)

    # Sanitize user-specified legend
    if legend is not None and not isinstance(legend, (list, tuple)):
        legend = [legend]

    # Make sure the viewers instances are consistent with the trajectories
    if viewers is None:
        viewers = [None for _ in trajs_data]
    assert len(viewers) == len(trajs_data)

    # Instantiate or refresh viewers if necessary
    for i, (viewer, traj, color) in enumerate(zip(
            viewers, trajs_data, robots_colors)):
        # Create new viewer instance if necessary, and load the robot in it
        if viewer is None:
            uniq_id = next(tempfile._get_candidate_names())
            robot = traj['robot']
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
                open_gui_if_parent=(record_video_path is None))
            viewers[i] = viewer

        # Reset robot model in viewer if requested color has changed
        if color is not None and color != viewer.robot_color:
            viewer._setup(traj['robot'], color)

    # Add default legend with robots names if replaying multiple trajectories
    if all(color is not None for color in robots_colors) and legend is None:
        legend = [viewer.robot_name for viewer in viewers]

    # Use first viewers as main viewer to call static methods conveniently
    viewer = viewers[0]

    # Make sure clock is only enabled for panda3d backend
    if enable_clock and not Viewer.backend.startswith('panda3d'):
        logger.warn(
            "`enable_clock` is only available with 'panda3d' backend.")
        enable_clock = False

    # Early return if nothing to replay
    if all(not len(traj['evolution_robot']) for traj in trajs_data):
        return viewers

    # Enable camera motion if requested
    if camera_motion is not None:
        Viewer.register_camera_motion(camera_motion)

    # Handle meshcat-specific options
    if legend is not None:
        Viewer.set_legend(legend)

    # Add watermark if requested
    if watermark_fullpath is not None:
        Viewer.set_watermark(watermark_fullpath)

    # Initialize robot configuration is viewer before any further processing
    for viewer_i, traj, offset in zip(viewers, trajs_data, xyz_offsets):
        data = traj['evolution_robot']
        if data:
            i = bisect_right([s.t for s in data], time_interval[0])
            viewer_i.display(data[i].q, data[i].v, offset)
        if Viewer.backend.startswith('panda3d'):
            if display_com is not None:
                viewer_i.display_center_of_mass(display_com)
            if display_dcm is not None:
                viewer_i.display_capture_point(display_dcm)
            if display_contacts is not None:
                viewer_i.display_contact_forces(display_contacts)
            if display_f_external is not None:
                viewer_i.display_external_forces(display_f_external)

    # Set camera pose or activate camera travelling if requested
    if travelling_frame is not None:
        viewer.attach_camera(travelling_frame, camera_xyzrpy)
    elif camera_xyzrpy is not None and any(camera_xyzrpy):
        viewer.set_camera_transform(*camera_xyzrpy)

    # Wait for the meshes to finish loading if video recording is disable
    if record_video_path is None:
        if Viewer.backend == 'meshcat':
            if verbose and not interactive_mode():
                print("Waiting for meshcat client in browser to connect: "
                      f"{Viewer._backend_obj.gui.url()}")
            Viewer.wait(require_client=True)
            if verbose and not interactive_mode():
                print("Browser connected! Replaying simulation...")

    # Handle start-in-pause mode
    if start_paused and record_video_path is None and not interactive_mode():
        input("Press Enter to continue...")
        if not Viewer.is_alive():
            return viewers

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
        position_evolutions, velocity_evolutions, force_evolutions = [], [], []
        for traj in trajs_data:
            if len(traj['evolution_robot']):
                data_orig = traj['evolution_robot']
                if traj['use_theoretical_model']:
                    model = traj['robot'].pinocchio_model_th
                else:
                    model = traj['robot'].pinocchio_model
                t_orig = np.array([s.t for s in data_orig])
                pos_orig = np.stack([s.q for s in data_orig], axis=0)
                position_evolutions.append(jiminy.interpolate(
                    model, t_orig, pos_orig, time_global))
                if data_orig[0].v is not None:
                    vel_orig = np.stack([s.v for s in data_orig], axis=0)
                    velocity_interp = interp1d(
                        t_orig,
                        vel_orig,
                        axis=0,
                        assume_sorted=True,
                        bounds_error=False,
                        fill_value=(vel_orig[0], vel_orig[-1]))
                    velocity_evolutions.append(velocity_interp(time_global))
                else:
                    velocity_evolutions.append((None,) * len(time_global))
                if data_orig[0].f_ext is not None:
                    forces = []
                    for i in range(len(data_orig[0].f_ext)):
                        f_ext_orig = np.stack([
                            s.f_ext[i] for s in data_orig], axis=0)
                        forces_interp = interp1d(
                            t_orig,
                            f_ext_orig,
                            axis=0,
                            assume_sorted=True,
                            bounds_error=False,
                            fill_value=(f_ext_orig[0], f_ext_orig[-1]))
                        forces.append(forces_interp(time_global))
                    force_evolutions.append([
                        [f_ext[i] for f_ext in forces]
                        for i in range(len(time_global))])
                else:
                    force_evolutions.append((None,) * len(time_global))
            else:
                position_evolutions.append(None)
                velocity_evolutions.append(None)
                force_evolutions.append(None)

        # Disable framerate limit of Panda3d for efficiency
        if Viewer.backend.startswith('panda3d'):
            framerate = viewer._backend_obj._app.get_framerate()
            viewer._backend_obj._app.set_framerate(None)

        # Initialize video recording
        if Viewer.backend == 'meshcat':
            # Sanitize the recording path to enforce '.webm' extension
            record_video_path = str(
                pathlib.Path(record_video_path).with_suffix('.webm'))

            # Start backend recording thread
            viewer._backend_obj.start_recording(
                VIDEO_FRAMERATE, *VIDEO_SIZE)
        else:
            # Sanitize the recording path to enforce '.mp4' extension
            record_video_path = str(
                pathlib.Path(record_video_path).with_suffix('.mp4'))

            # Create ffmpeg video writer
            out = av.open(record_video_path, mode='w')
            out.metadata['title'] = scene_name
            stream = out.add_stream('libx264', rate=VIDEO_FRAMERATE)
            stream.width, stream.height = VIDEO_SIZE
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = VIDEO_QUALITY * (8 * 1024 ** 2)

        # Add frames to video sequentially
        for i, t_cur in enumerate(tqdm(
                time_global, desc="Rendering frames", disable=(not verbose))):
            # Update 3D view
            for viewer, pos, vel, forces, xyz_offset, update_hook in zip(
                    viewers, position_evolutions, velocity_evolutions,
                    force_evolutions, xyz_offsets, update_hooks):
                if pos is None:
                    continue
                q, v, f_ext = pos[i], vel[i], forces[i]
                if f_ext is not None:
                    for i, f_ext in enumerate(f_ext):
                        viewer.f_external[i].vector[:] = f_ext
                if update_hook is not None:
                    update_hook_t = partial(update_hook, t_cur, q, v)
                else:
                    update_hook_t = None
                viewer.display(q, v, xyz_offset, update_hook_t)

            # Update clock if enabled
            if enable_clock:
                Viewer.set_clock(t_cur)

            # Add frame to video
            if Viewer.backend == 'meshcat':
                viewer._backend_obj.add_frame()
            else:
                # Capture frame
                frame = viewer.capture_frame(*VIDEO_SIZE)

                # Write frame
                frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                for packet in stream.encode(frame):
                    out.mux(packet)

        # Finalize video recording
        if Viewer.backend == 'meshcat':
            # Stop backend recording thread
            viewer._backend_obj.stop_recording(record_video_path)
        else:
            # Flush and close recording file
            for packet in stream.encode(None):
                out.mux(packet)
            out.close()

        # Restore framerate limit of Panda3d
        if Viewer.backend.startswith('panda3d'):
            viewer._backend_obj._app.set_framerate(framerate)
    else:
        # Play trajectories with multithreading
        def replay_thread(viewer, *args):
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
                      enable_clock)))
        for thread in threads:
            thread.daemon = True
            thread.start()
        for thread in threads:
            thread.join()

    if Viewer.is_alive():
        # Disable camera travelling and camera motion if it was enabled
        if travelling_frame is not None:
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

    return viewers


def extract_replay_data_from_log_data(
        robot: jiminy.Robot,
        log_data: Dict[str, np.ndarray]) -> Tuple[
            TrajectoryDataType, Callable[[float], None], Any]:
    """Extract replay data from log data.

    :param robot: Jiminy robot for which to extract log data.
    :param log_data: Data from the log file, in a dictionnary.

    :returns: Trajectory data, update hook and extra keyword arguments to
              forward to `play_trajectories` method to display the trajectory.
              By default, it enables display of external forces applied on
              freeflyer if any.
    """
    # For each pair (log, robot), extract a trajectory object for
    # `play_trajectories`
    trajectory = extract_trajectory_data_from_log(log_data, robot)

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
        update_hook = emulate_sensors_data_from_log(log_data, robot)
    else:
        logger.warn(
            "At least one of the robot is locked, which means that a "
            "simulation using the robot is still running. It will be "
            "impossible to display sensor data. Call `simulator.stop` to "
            "unlock the robot before replaying logs data.")
        update_hook = None

    return trajectory, update_hook, replay_kwargs


def play_logs_data(robots: Union[Sequence[jiminy.Robot], jiminy.Robot],
                   logs_data: Union[Sequence[Dict[str, np.ndarray]],
                                    Dict[str, np.ndarray]],
                   **kwargs) -> Sequence[Viewer]:
    """Play log data in a viewer.

    This method simply formats the data then calls `play_trajectories`.

    :param robots: Either a single robot, or a list of robot for each log data.
    :param logs_data: Either a single dictionary, or a list of dictionaries of
                      simulation data log.
    :param kwargs: Keyword arguments to forward to `play_trajectories` method.
    """
    # Reformat input arguments as lists
    if not isinstance(logs_data, (list, tuple)):
        logs_data = [logs_data]
    if not isinstance(robots, (list, tuple)):
        robots = [robots]

    # Extract a replay data for `play_trajectories` for each pair (robot, log)
    trajectories, update_hooks, extra_kwargs = [], [], {}
    for robot, log_data in zip(robots, logs_data):
        traj, update_hook, _kwargs = extract_replay_data_from_log_data(
            robot, log_data)
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
                    mesh_package_dirs: Union[str, Sequence[str]] = (),
                    **kwargs) -> Sequence[Viewer]:
    """Play the content of a logfile in a viewer.

    This method reconstruct the exact model used during the simulation
    corresponding to the logfile, including random biases or flexible joints.

    :param logs_files: Either a single simulation log files in any format, or
                       a list.
    :param mesh_package_dirs: Prepend custom mesh package seach path
                              directories to the ones provided by log file. It
                              may be necessary to specify it to read log
                              generated on a different environment.
    :param kwargs: Keyword arguments to forward to `play_trajectories` method.
    """
    # Reformat as list
    if not isinstance(logs_files, (list, tuple)):
        logs_files = [logs_files]

    # Extract log data and build robot for each log file
    robots, logs_data = [], []
    for log_file in logs_files:
        log_data, log_constants = read_log(log_file)
        robot = build_robot_from_log_constants(
            log_constants, mesh_package_dirs)
        logs_data.append(log_data)
        robots.append(robot)

    # Default legend if several log files are provided
    if "legend" not in kwargs and len(logs_files) > 1:
        kwargs["legend"] = [pathlib.Path(log_file).stem.split('_')[-1]
                            for log_file in logs_files]

    # Forward arguments to lower-level method
    return play_logs_data(robots, logs_data, **kwargs)


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
        help=("Whether or not to track the root frame of the first robot, "
              "assuming the robot has a freeflyer."))
    parser.add_argument(
        '-b', '--backend', default='panda3d',
        help="Display backend (panda3d, meshcat, or gepetto-gui).")
    parser.add_argument(
        '-m', '--mesh_package_dir', default=None,
        help="Fullpath location of mesh package directory.")
    parser.add_argument(
        '-v', '--record_video_path', default=None,
        help="Fullpath location where to save generated video.")
    options, files = parser.parse_known_args()
    kwargs = vars(options)
    kwargs['logs_files'] = files

    # Convert mesh package dir into a list
    if kwargs['mesh_package_dir'] is not None:
        kwargs['mesh_package_dirs'] = [kwargs.pop('mesh_package_dir')]

    # Convert travelling mode into frame index
    if kwargs.pop('travelling'):
        kwargs['travelling_frame'] = 2

    # Replay trajectories
    repeat = True
    viewers = None
    while repeat:
        viewers = play_logs_files(**{**dict(
            remove_widgets_overlay=False,
            viewers=viewers),
            **kwargs})
        kwargs["start_paused"] = False
        if not hasattr(kwargs, "camera_xyzrpy"):
            kwargs["camera_xyzrpy"] = None
        if kwargs["record_video_path"] is None:
            while True:
                reply = input("Do you want to replay again (y/[n])?").lower()
                if not reply or reply in ("y", "n"):
                    break
            repeat = (reply == "y")
        else:
            repeat = False

    # Do not exit method as long as a graphical window is open
    while Viewer.has_gui():
        time.sleep(0.5)
