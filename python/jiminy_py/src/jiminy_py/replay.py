import logging
import pathlib
import asyncio
import tempfile
from threading import Thread, Lock
from itertools import cycle, islice
from typing import Optional, Union, Sequence, Dict

import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from typing_extensions import TypedDict

from . import core as jiminy
from .state import State
from .viewer import (
    Viewer, Tuple3FType, Tuple4FType, CameraPoseType, CameraMotionType)
from .meshcat.utilities import interactive_mode


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


class TrajectoryDataType(TypedDict, total=False):
    # List of State objects of increasing time.
    evolution_robot: Sequence[State]
    # Jiminy robot. None if omitted.
    robot: Optional[jiminy.Robot]
    # Whether to use theoretical or actual model
    use_theoretical_model: bool


ColorType = Union[Tuple4FType, str]


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
        qe = np.stack([log_data[".".join(("HighLevelController", s))]
                       for s in robot.logfile_position_headers], axis=-1)
    except KeyError:
        model_options['dynamics']['enableFlexibleModel'] = \
            not robot.is_flexible
        robot.set_model_options(model_options)
        qe = np.stack([log_data[".".join(("HighLevelController", s))]
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
        Viewer.register_camera_motion(camera_motion)

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