from pathlib import Path

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

import jiminy_py.core as jiminy
import jiminy_py.viewer.panda3d.panda3d_visualizer as panda3d_viewer
import jiminy_py.viewer.replay as replay
from jiminy_py.core import EncoderSensor as encoder
from jiminy_py.viewer import Viewer, play_trajectories
from jiminy_py.dynamics import State, update_quantities
from jiminy_py.log import update_sensors_data_from_log
from jiminy_py.robot import load_hardware_description_file
import pinocchio as pin

from wdc.telemetry import LogFile
from wdc.telemetry.tools.tlmc_crop import tlmc_crop_logfile
from wdc.telemetry.extractors.extraction import extract_telemetry
from wdc.logstudio.logstudio import build_logstudio_data
from wdc.logstudio.viewer3d_widget import RobotData
from wdc.simu_jiminy.domain.ground import GroundConfiguration


replay.VIDEO_FRAMERATE = 60
replay.VIDEO_QUALITY = 2.0

log_dir = Path("/home/builder/wdc_workspace/logs/keynote/")
# log_data_path = log_dir / "20231113T161933Z_LogFile.data.0"
log_data_path = log_dir / "20231113T161933Z_LogFile_downstairs.tlmc"
start_time = 1498.5 - 3.0
end_time = 1506.5 + 3.0
prefix = "HighLevelController.Admittance"
urdf = log_dir / "exo_empty.urdf"

FLEX_OBSERVER_COLOR = (0.0, 1.0, 0.0, 1.0)
DESIRED_OBSERVER_COLOR = (1.0, 1.0, 0.0, 1.0)
TARGET_OBSERVER_COLOR = (1.0, 0.0, 0.0, 1.0)

if log_data_path.suffix != ".tlmc":
    log_data_path = extract_telemetry(log_data_path)
    log_data = LogFile(log_data_path, cache=False)
    out_data = log_dir / f"{log_data_path.name.split('.', 1)[0]}_downstairs.tlmc"
    tlmc_crop_logfile(log_data, out_data, start_time=start_time, end_time=end_time)
    log_data_path = out_data
ldata = build_logstudio_data(log_data_path, urdf, workspace=log_dir / "wks")
assert ldata is not None

# rigid_robot_data = RobotData(
#     "measured",
#     str(urdf),
#     TARGET_OBSERVER_COLOR,
#     use_flexible=False,
#     flexibility_centers=ldata.product.flexibilities,
# )
# rigid_robot_data.load_data(
#     ldata.admittance_loader,
#     controller_prefix=ldata.controller.prefix,
#     common_variable_name="measuredModelPosition",
#     use_odometry=True,
# )
# desired_robot_data = RobotData(
#     "measured",
#     str(urdf),
#     DESIRED_OBSERVER_COLOR,
#     use_flexible=False,
#     flexibility_centers=ldata.product.flexibilities,
# )
# desired_robot_data.load_data(
#     ldata.admittance_loader,
#     controller_prefix=ldata.controller.prefix,
#     common_variable_name="desiredModelPosition",
#     use_odometry=True,
# )
flex_robot_data = RobotData(
    "flexible",
    str(urdf),
    FLEX_OBSERVER_COLOR,
    use_flexible=True,
    flexibility_centers=ldata.product.flexibilities,
    project_on_ground=False,
)
flex_robot_data.load_data(
    ldata.admittance_loader,
    controller_prefix=ldata.controller.prefix,
    common_variable_name="flexibleObservedWorldModelPosition",
    use_odometry=False,
)
for robot_data in (
        # desired_robot_data,
        # rigid_robot_data,
        flex_robot_data,
    ):
    oMf_list = []
    for i in (-424, -422):
        update_quantities(robot_data.robot,
                          robot_data.q[i],
                          use_theoretical_model=False)
        frame_idx = robot_data.robot.pinocchio_model.getFrameId("RightSole")
        oMf_list.append(robot_data.robot.pinocchio_data.oMf[frame_idx].copy())
    oMd = oMf_list[0] * oMf_list[1].inverse()
    for j in range(i, 0):
        q_j = robot_data.q[j]
        T = oMd * pin.SE3(pin.Quaternion(q_j[3:7]).matrix(), q_j[:3])
        q_j[:3], q_j[3:7] = T.translation, pin.Quaternion(T.rotation).coeffs()

    for name in ("LeftSittingPoint", "RightSittingPoint"):
        if not robot_data.robot.pinocchio_model.existFrame(name):
            robot_data.robot.add_frame(name, "PelvisLink", pin.SE3.Identity())
    robot_data.robot.remove_contact_points(robot_data.robot.contact_frames_names)
    robot_data.robot.detach_sensors()
    robot_data.robot.detach_motors()
    load_hardware_description_file(
        robot_data.robot,
        "/src/pocs/simu_jiminy/data/eve_beta/hardware.toml",
        avoid_instable_collisions=False,
        verbose=False)
    robot_data.robot.detach_sensors('ForceSensor')
    for frame_name in (
        "LeftInternalHeel",  "LeftExternalHeel", "LeftExternalToe", "LeftInternalToe",
        "RightInternalHeel", "RightExternalHeel", "RightExternalToe", "RightInternalToe"
    ):
        robot_data.robot.add_contact_points([frame_name,])
        sensor_name = ''.join(e for e in frame_name if e.isupper())
        sensor = jiminy.ContactSensor(sensor_name)
        robot_data.robot.attach_sensor(sensor)
        sensor.initialize(frame_name)

log_data = {"Global.Time": robot_data.t - robot_data.t[0]}
for name in (
        "LEH", "LET", "LIH", "LIT", "REH", "RET", "RIH", "RIT"
    ):
    var = ldata.admittance_loader.get(f"ForceSensors.{name}.Fz.value")
    for field in ("FX", "FY"):
        log_data[".".join((name, field))] = np.zeros_like(robot_data.t)
    fq, fs = 5.0, 1000.0
    b, a = signal.butter(2, fq / (fs / 2))
    value_filt = signal.filtfilt(b, a, var.value)
    log_data[f"{name}.FZ"] = interp1d(var.time, value_filt)(robot_data.t)
for name in (
        "PelvisIMU", "LeftThighLowerIMU", "LeftThighUpperIMU",
        "LeftTibiaIMU", "LeftFootIMU", "RightThighLowerIMU",
        "RightThighUpperIMU", "RightTibiaIMU", "RightFootIMU",
    ):
    for axis in ("x", "y", "z", "w"):
        log_data[f"{name}.Quat{axis}"] = np.zeros_like(robot_data.t)
    for axis in ("x", "y", "z"):
        for data_type in ("Gyro", "Accel"):
            var = ldata.admittance_loader.get(f"{name}.{data_type.lower()}{axis.upper()}")
            fq, fs = 7.0, 1000.0
            b, a = signal.butter(2, fq / (fs / 2))
            value_filt = signal.filtfilt(b, a, var.value)
            log_data[f"{name}.{data_type}{axis}"] = interp1d(var.time, value_filt)(robot_data.t)

for side, camera_pose in (
        # ("front", (np.array([4.50, -0.84, 1.31]), np.array([1.47, 0.0, 1.4]))),
        # ("side", (np.array([1.12, 4.75, 1.18]), np.array([1.53, 0.0,  3.0]))),
        ("3_4", ((np.array([3.90, 1.84, 1.66]), np.array([1.38, 0.0, 2.07])))),
        ):
    for robot_name, robot_data in (
            # ("desired", desired_robot_data),
            # ("rigid", rigid_robot_data),
            ("flex", flex_robot_data),
            ):
        panda3d_viewer.SKY_TOP_COLOR = (0.055, 0.075, 0.17, 1.0)
        panda3d_viewer.SKY_BOTTOM_COLOR = (0.055, 0.075, 0.17, 1.0)

        Viewer.close()
        Viewer.connect_backend("panda3d") #-sync")
        Viewer.set_camera_transform(None, *camera_pose)

        gui = Viewer._backend_obj.gui
        gui.append_group("world")
        gui.append_box("world", "stair", (1.0, 1.0, 0.11))
        gui.move_node("world", "stair", ((-0.22, -0.1, 0.06), ((
            quat := pin.Quaternion(pin.rpy.rpyToMatrix(0.0, 0.0, -0.12)).coeffs()
            )[-1], *quat[:-1])))
        gui.set_material("world", "stair", (0.48, 0.51, 0.87, 1.0))
        gui.show_axes(False)
        gui.show_floor(False)

        update_hook = update_sensors_data_from_log(log_data, robot_data.robot)

        encoder_poses = {}
        for name in robot_data.robot.sensors_names[encoder.type]:
            sensor = robot_data.robot.get_sensor(encoder.type, name)
            joint_idx, data = sensor.joint_idx, sensor.data
            joint = robot_data.robot.pinocchio_model.joints[joint_idx]
            oMi = robot_data.robot.pinocchio_data.oMi[joint_idx]
            axis = ord(joint.shortname()[-1]) - ord('X')
            encoder_poses[name] = (
                oMi.translation, oMi.rotation, axis, joint.idx_q)

        offsets = {
            'RightFrontalHipJoint': 0.06,
            'RightTransverseHipJoint': 0.08,
            'RightSagittalHipJoint': 0.16,
            'RightSagittalKneeJoint': 0.16,
            'RightSagittalAnkleJoint': 0.14,
            'RightFrontalAnkleJoint': 0.12,
            'LeftFrontalHipJoint': 0.06,
            'LeftTransverseHipJoint': 0.08,
            'LeftSagittalHipJoint': -0.16,
            'LeftSagittalKneeJoint': -0.16,
            'LeftSagittalAnkleJoint': -0.14,
            'LeftFrontalAnkleJoint': 0.12
        }

        def update_hook_full(t: float, q: np.ndarray, v: np.ndarray) -> None:
            update_hook(t, q, v)

            move_pose_dict = {}
            for name, (pos, rot, axis, idx_q) in encoder_poses.items():
                q_i = 2.0 * q[idx_q]
                if q_i > 0.0:
                    theta_start, theta_end = 0.0, q_i
                else:
                    theta_start, theta_end = 2.0 * np.pi + q_i, 2.0 * np.pi

                gui.remove_node("world", name)
                gui.append_cylinder("world",
                                    name,
                                    radius=0.05,
                                    length=0.005,
                                    theta_start=theta_start,
                                    theta_end=theta_end,
                                    anchor_bottom=True)
                gui.show_node("world", name, True, always_foreground=True)
                gui.set_material(
                    "world", name, (1.5 * 0.48, 1.5 * 0.51, 1.5 * 0.87, 1.0))

                if axis == 0:
                    rot = np.stack((rot[:, 2], rot[:, 1], -rot[:, 0]), axis=1)
                elif axis == 1:
                    rot = np.stack((rot[:, 0], rot[:, 2],-rot[:, 1]), axis=1)
                quat = pin.Quaternion(rot).coeffs()
                move_pose_dict[name] = (
                    pos + offsets[name] * rot[:, 2], (quat[-1], *quat[:-1]))
            gui.move_nodes("world", move_pose_dict)

        play_trajectories(
            trajs_data=({
                "evolution_robot": [
                    State(t - robot_data.t[0], q)
                    for t, q in zip(robot_data.t, robot_data.q)],
                "robot": robot_data.robot,
                "use_theoretical_model": False
            },),
            update_hooks=(update_hook_full,),
            # robots_colors=(robot_data.color_rgba,),
            robots_colors=((0.5, 0.5, 0.5, 1.0),),
            display_com=False,
            display_dcm=False,
            display_contacts=True,
            display_imus=True,
            display_f_external=False,
            delete_robot_on_close=False,
            # record_video_path=f"{robot_name}_robot_{side}.mp4",
            record_video_size=(2704, 1520)
        )

