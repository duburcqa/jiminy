import time
import logging
import pathlib
import asyncio
import tempfile
import argparse
from bisect import bisect_right
from functools import partial
from threading import Thread, Lock
from itertools import cycle, islice
from typing import Optional, Union, Sequence, Tuple, Dict, Any, Callable

import av
import numpy as np
from tqdm import tqdm

import pinocchio as pin

from .. import core as jiminy
from ..log import (TrajectoryDataType,
                   build_robot_from_log,
                   extract_viewer_data_from_log,
                   emulate_sensors_data_from_log)
from .viewer import (
    COLORS, Viewer, Tuple3FType, Tuple4FType, CameraPoseType, CameraMotionType)
from .meshcat.utilities import interactive_mode


VIDEO_FRAMERATE = 30
VIDEO_SIZE = (800, 800)
VIDEO_QUALITY = 0.3  # [Mbytes/s]


logger = logging.getLogger(__name__)


ColorType = Union[Tuple4FType, str]


def play_trajectories(trajectory_data: Union[
                          TrajectoryDataType, Sequence[TrajectoryDataType]],
                      update_hooks: Optional[Union[
                          Callable[[float], None],
                          Sequence[Callable[[float], None]]]] = None,
                      time_interval: Optional[Union[
                          np.ndarray, Tuple[float, float]]] = (0.0, np.inf),
                      speed_ratio: float = 1.0,
                      xyz_offsets: Optional[Union[
                          Tuple3FType, Sequence[Tuple3FType]]] = None,
                      robots_colors: Optional[Union[
                          ColorType, Sequence[ColorType]]] = None,
                      travelling_frame: Optional[str] = None,
                      camera_xyzrpy: Optional[CameraPoseType] = None,
                      camera_motion: Optional[CameraMotionType] = None,
                      watermark_fullpath: Optional[str] = None,
                      legend: Optional[Union[str, Sequence[str]]] = None,
                      enable_clock: bool = False,
                      scene_name: str = 'world',
                      record_video_path: Optional[str] = None,
                      start_paused: bool = False,
                      backend: Optional[str] = None,
                      delete_robot_on_close: Optional[bool] = None,
                      remove_widgets_overlay: bool = True,
                      close_backend: bool = False,
                      viewers: Sequence[Viewer] = None,
                      verbose: bool = True,
                      **kwargs: Any) -> Sequence[Viewer]:
    """Replay one or several robot trajectories in a viewer.

    The ratio between the replay and the simulation time is kept constant to
    the desired ratio. One can choose between several backend (gepetto-gui or
    meshcat).

    .. note::
        Replay speed is independent of the platform (windows, linux...) and
        available CPU power.

    :param trajectory_data: List of `TrajectoryDataType` dicts.
    :param update_hooks: Callables associated with each robot that can be used
                         to update non-kinematic robot data, for instance to
                         emulate sensors data from log using the hook provided
                         by `emulate_sensors_data_from_log` method. None` to
                         disable.
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
                         Only available with panda3d rendering backend.
                         Optional: Disable by default.
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
    :param backend: Backend, one of 'meshcat' or 'gepetto-gui'. If None,
                    'meshcat' is used in notebook environment and 'gepetto-gui'
                    otherwise.
                    Optional: None by default.
    :param delete_robot_on_close: Whether or not to delete the robot from the
                                  viewer when closing it.
                                  Optional: True by default.
    :param remove_widgets_overlay: Remove overlay (legend, watermark, clock,
                                   ...) automatically before returning.
                                   Optional: Enable by default.
    :param close_backend: Whether or not to close backend automatically before
                          returning.
                          Optional: Disable by default.
    :param viewers: List of already instantiated viewers, associated one by one
                    in order to each trajectory data. None to disable.
                    Optional: None by default.
    :param verbose: Add information to keep track of the process.
                    Optional: True by default.
    :param kwargs: Used argument to allow chaining renderining methods.

    :returns: List of viewers used to play the trajectories.
    """
    # Make sure trajectory data and update hook are list or tuple
    if not isinstance(trajectory_data, (list, tuple)):
        trajectory_data = [trajectory_data]
    if not isinstance(update_hooks, (list, tuple)):
        update_hooks = [update_hooks]

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

    # Sanitize user-specified robot offsets
    if xyz_offsets is None:
        xyz_offsets = len(trajectory_data) * [None]
    elif len(xyz_offsets) != len(trajectory_data):
        xyz_offsets = np.tile(xyz_offsets, (len(trajectory_data), 1))

    # Sanitize user-specified robot colors
    if robots_colors is None:
        if len(trajectory_data) == 1:
            robots_colors = [None]
        else:
            robots_colors = list(islice(
                cycle(COLORS.values()), len(trajectory_data)))
    elif not isinstance(robots_colors, (list, tuple)) or \
            isinstance(robots_colors[0], float):
        robots_colors = [robots_colors]
    assert len(robots_colors) == len(trajectory_data)

    # Sanitize user-specified legend
    if legend is not None and not isinstance(legend, (list, tuple)):
        legend = [legend]

    # Instantiate or refresh viewers if necessary
    if viewers is None:
        # Delete robot by default only if not in notebook
        if delete_robot_on_close is None:
            delete_robot_on_close = not interactive_mode()

        # Create new viewer instances
        viewers = []
        lock = Lock()
        uniq_id = next(tempfile._get_candidate_names())
        for i, (traj, color) in enumerate(zip(trajectory_data, robots_colors)):
            # Create a new viewer instance, and load the robot in it
            robot = traj['robot']
            robot_name = f"{uniq_id}_robot_{i}"
            use_theoretical_model = traj['use_theoretical_model']
            viewer = Viewer(
                robot,
                use_theoretical_model=use_theoretical_model,
                robot_color=color,
                robot_name=robot_name,
                lock=lock,
                backend=backend,
                scene_name=scene_name,
                delete_robot_on_close=delete_robot_on_close,
                open_gui_if_parent=(record_video_path is None))
            viewers.append(viewer)
    else:
        # Reset robot model in viewer if requested color has changed
        for viewer, traj, color in zip(
                viewers, trajectory_data, robots_colors):
            if color != viewer.robot_color:
                viewer._setup(traj['robot'], color)
    assert len(viewers) == len(trajectory_data)

    # Add default legend with robots names if replaying multiple trajectories
    if all(color is not None for color in robots_colors) and legend is None:
        legend = [viewer.robot_name for viewer in viewers]

    # Use first viewers as main viewer to call static methods conveniently
    viewer = viewers[0]

    # Make sure clock is only enabled for panda3d backend
    if enable_clock and Viewer.backend != 'panda3d':
        logger.warn(
            "`enable_clock` is only available with 'panda3d' backend.")
        enable_clock = False

    # Early return if nothing to replay
    if all(not len(traj['evolution_robot']) for traj in trajectory_data):
        return viewers

    # Set camera pose or activate camera travelling if requested
    if travelling_frame is not None:
        viewer.attach_camera(travelling_frame, camera_xyzrpy)
    elif camera_xyzrpy is not None:
        viewer.set_camera_transform(*camera_xyzrpy)

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
    for viewer_i, traj, offset in zip(viewers, trajectory_data, xyz_offsets):
        evolution_robot = traj['evolution_robot']
        if len(evolution_robot):
            i = bisect_right([s.t for s in evolution_robot], time_interval[0])
            viewer_i.display(evolution_robot[i].q, offset)

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

    # Replay the trajectory
    if record_video_path is not None:
        # Extract and resample trajectory data at fixed framerate
        time_max = time_interval[0]
        for traj in trajectory_data:
            if len(traj['evolution_robot']):
                time_max = max([time_max, traj['evolution_robot'][-1].t])
        time_max = min(time_max, time_interval[1])

        time_global = np.arange(
            time_interval[0], time_max, speed_ratio / VIDEO_FRAMERATE)
        position_evolutions = []
        for traj in trajectory_data:
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
            else:
                position_evolutions.append(None)

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
            # Update the configurations of the robots
            for viewer, positions, xyz_offset, update_hook in zip(
                    viewers, position_evolutions, xyz_offsets, update_hooks):
                if positions is not None:
                    viewer.display(
                        positions[i], xyz_offset, partial(update_hook, t_cur))

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
        def replay_thread(viewer, *args):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            viewer.replay(*args)

        # Play trajectories with multithreading
        threads = []
        for viewer, traj, xyz_offset, update_hook in zip(
                viewers, trajectory_data, xyz_offsets, update_hooks):
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


def play_logs_data(robots: Union[Sequence[jiminy.Robot], jiminy.Robot],
                   logs_data: Union[Sequence[Dict[str, np.ndarray]],
                                    Dict[str, np.ndarray]],
                   **kwargs) -> Sequence[Viewer]:
    """Play log data in a viewer.

    This method simply formats the data then calls play_trajectories.

    :param robots: Either a single robot, or a list of robot for each log data.
    :param logs_data: Either a single dictionary, or a list of dictionaries of
                      simulation data log.
    :param kwargs: Keyword arguments to forward to `play_trajectories` method.
    """
    # Reformat input arguments as lists
    if not isinstance(logs_data, (list, tuple)):
        logs_data = [logs_data]
    if not isinstance(robots, (list, tuple)):
        robots = [robots] * len(logs_data)

    # For each pair (log, robot), extract a trajectory object for
    # `play_trajectories`
    trajectories = [extract_viewer_data_from_log(log, robot)
                    for log, robot in zip(logs_data, robots)]

    # Define `update_hook` to emulate sensor update
    update_hooks = [emulate_sensors_data_from_log(log, robot)
                    for log, robot in zip(logs_data, robots)]

    # Finally, play the trajectories
    return play_trajectories(trajectories, update_hooks, **kwargs)


def play_logs_files(logs_files: Union[str, Sequence[str]],
                    **kwargs) -> Sequence[Viewer]:
    """Play the content of a logfile in a viewer.

    This method reconstruct the exact model used during the simulation
    corresponding to the logfile, including random biases or flexible joints.

    :param logs_files: Either a single simulation log files in any format, or
                       a list.
    :param kwargs: Keyword arguments to forward to `play_trajectories` method.
    """
    # Reformat as list
    if not isinstance(logs_files, (list, tuple)):
        logs_files = [logs_files]

    # Extract log data and build robot for each log file
    robots, logs_data = [], []
    for log_file in logs_files:
        robot, (log_data, _) = build_robot_from_log(log_file)
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
        '-s', '--speed_ratio', type=float, default=0.5,
        help="Real time to simulation time factor.")
    parser.add_argument(
        '-b', '--backend', default='panda3d',
        help="Display backend (panda3d, meshcat, or gepetto-gui).")
    options, files = parser.parse_known_args()
    kwargs = vars(options)
    kwargs['logs_files'] = files

    # Replay trajectories
    play_logs_files(**{"remove_widgets_overlay": False, **kwargs})

    # Do not exit method as long as Jiminy viewer is open
    while Viewer.is_alive():
        time.sleep(0.5)
