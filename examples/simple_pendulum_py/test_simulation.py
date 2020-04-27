import os
import tempfile
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt


from jiminy_py import core as jiminy
from jiminy_py.log import extract_viewer_data_from_log
from jiminy_py.viewer import play_trajectories
from jiminy_py.core import HeatMapFunctor, heatMapType_t, ForceSensor
import pinocchio as pin

from interactive_plot_util import interactive_legend


SPATIAL_COORDS = ["X", "Y", "Z"]


# ################################ User parameters #######################################
# Parse arguments.
parser = argparse.ArgumentParser(description="Compute and plot inverted pendulum solutions")
parser.add_argument('-tf', '--tf', type=float, help="Solve duration.", default=2.0)
parser.add_argument('-fHLC', '--fHLC', type=float, help="HLC frequence (LLC = 1kHz).", default=200.0)
parser.add_argument('--acceleration', help='Command directly with acceleration (default position control).', action='store_true', default=False)
parser.add_argument('--plot', help='Plot.', action='store_true', default=False)
parser.add_argument('--show', help='Show gepetto animation.', action='store_true', default=False)
parser.add_argument('--targetsFB', help='Target state instead of current in com, dcm and zmp computation.', action='store_true', default=False)
parser.add_argument('--mixedFB', help='Current COM and target VCOM states for DCM computation.', action='store_true', default=False)
parser.add_argument('--clampCmd', help='Clamp zmp command.', action='store_true', default=False)
args = parser.parse_args()

tf = args.tf
fHLC = args.fHLC
acceleration_control = args.acceleration
position_control = not acceleration_control

script_dir = os.path.dirname(os.path.realpath(__file__))
os.environ["JIMINY_MESH_PATH"] = os.path.join(script_dir, "../../data")
urdf_path = os.path.join(os.environ["JIMINY_MESH_PATH"], "simple_pendulum/simple_pendulum.urdf")

# ########################### Initialize the simulation #################################

# Instantiate the robot
contact_points = ["Corner1", "Corner2", "Corner3", "Corner4"]
motor_joint_names = ("PendulumJoint",)
force_sensor_def = {"F1": "Corner1",
                    "F2": "Corner2",
                    "F3": "Corner3",
                    "F4": "Corner4"}

robot = jiminy.Robot()
robot.initialize(urdf_path, True)
for joint_name in motor_joint_names:
    motor = jiminy.SimpleMotor(joint_name)
    robot.attach_motor(motor)
    motor.initialize(joint_name)
for sensor_name, frame_name in force_sensor_def.items():
    force_sensor = jiminy.ForceSensor(sensor_name)
    robot.attach_sensor(force_sensor)
    force_sensor.initialize(frame_name)
robot.add_contact_points(contact_points)

# Extract some constant
iPos = robot.motors_position_idx[0]
iVel = robot.motors_velocity_idx[0]
axisCom = 1

# Constants
m = 75
l = 1.0
g  = 9.81
omega = np.sqrt(g/l)

# Initial values
q0 = 0.0
dq0 = 0.0
x0 = np.zeros((robot.nq + robot.nv, ))
x0[:robot.nq] = pin.neutral(robot.pinocchio_model_th)
x0[iPos] = q0
x0[iPos+iVel] = dq0

# Compute com dcm references
nTimes = int(tf * 1.0e3) + 1
deltaStabilization = 0.5e3
deltaSlope = 1.0
deltaCom = 0.041
comRef = np.zeros(nTimes)
comRef[int(deltaStabilization):(int(deltaStabilization + deltaSlope * 1.0e3) + 1)] = \
    np.linspace(0, deltaCom, int(deltaSlope * 1.0e3) + 1, endpoint=False)
comRef[(int(deltaStabilization + deltaSlope * 1.0e3) + 1):] = deltaCom
zmpRef = comRef
dcomRef = np.zeros(nTimes)
dcomRef[int(deltaStabilization):(int(deltaStabilization + deltaSlope * 1.0e3) + 1)] = deltaCom / deltaSlope
ddcomRef = np.zeros(nTimes)

if args.targetsFB:
    # Gains dcm control
    Kpdcm = 15.0
    Kddcm = 2.0
    Kidcm = 1.0
    decay = 0.1
    integral_ = 0.0
    # Gains admittance
    Acom = 15.0
elif args.mixedFB:
    # Gains dcm control
    Kpdcm = 15.0
    Kddcm = 1.0
    Kidcm = 0.0
    decay = 0.0
    integral_ = 0.0
    # Gains admittance
    Acom = 7.5
else:
    # Gains dcm control
    Kpdcm = 1.0
    Kddcm = 0.0
    Kidcm = 0.5
    decay = 0.01
    integral_ = 0.0
    # Gains admittance
    Acom = 60.0
# Gains position control
Kp = (m * (l ** 2)) * 1.0e3
Kd = 0.0 * Kp

# Perturbation
paux = 0.02
taux = 6.0

# Utilities
def update_frame(robot, data, name):
    frame_id = robot.getFrameId(name)
    pin.updateFramePlacement(robot, data, frame_id)

def get_frame_placement(robot, data, name):
    frame_id = robot.getFrameId(name)
    return data.oMf[frame_id]

# Logging: create global variables to make sure they never get deleted
com = pin.centerOfMass(robot.pinocchio_model_th, robot.pinocchio_data_th, x0[:robot.nq], x0[robot.nq:])
vcom = robot.pinocchio_data_th.vcom[0]
dcm = com + vcom / omega
totalWrench = pin.Force.Zero()
zmp = np.zeros((2,))
zmp[axisCom] = zmpRef[0]
zmp_cmd = zmp.copy()
state_target = np.array([0.0, 0.0])

com_log, comTarget_log, comRef_log = com.copy(), com.copy(), com.copy()
vcom_log, vcomTarget_log, vcomRef_log = vcom.copy(), vcom.copy(), vcom.copy()
dcm_log, dcmTarget_log, dcmRef_log = dcm.copy(), dcm.copy(), dcm.copy()
totalWrench_angular_log = totalWrench.angular.copy()
totalWrench_linear_log = totalWrench.linear.copy()
zmp_log, zmpTarget_log, zmpRef_log = zmp.copy(), zmp.copy(), zmp.copy()
zmp_cmd_log = zmp.copy()
state_target_log = state_target.copy()

# Instantiate the controller
t_1 = 0.0
u_1 = 0.0
qi = np.zeros((robot.nq, ))
dqi = np.zeros((robot.nv, ))
ddqi = np.zeros((robot.nv, ))
def updateState(robot, q, v, sensor_data):
    # Get dcm from current state
    pin.forwardKinematics(robot.pinocchio_model_th, robot.pinocchio_data_th, q, v)
    comOut = pin.centerOfMass(robot.pinocchio_model_th, robot.pinocchio_data_th, q, v)
    vcomOut = robot.pinocchio_data_th.vcom[0]
    dcmOut = comOut + vcomOut / omega

    # Create zmp from forces
    forces = np.asarray(sensor_data[ForceSensor.type])
    newWrench = pin.Force.Zero()
    for i,name in enumerate(contact_points):
        update_frame(robot.pinocchio_model_th, robot.pinocchio_data_th, name)
        placement = get_frame_placement(robot.pinocchio_model_th, robot.pinocchio_data_th, name)
        wrench = pin.Force(np.array([0.0, 0.0, forces[2, i]]), np.zeros(3))
        newWrench += placement.act(wrench)
    totalWrenchOut = newWrench
    if totalWrenchOut.linear[2] > 0:
        zmpOut = [-totalWrenchOut.angular[1] / totalWrenchOut.linear[2],
                   totalWrenchOut.angular[0] / totalWrenchOut.linear[2]]
    else:
        zmpOut = zmp_log

    return comOut, vcomOut, dcmOut, zmpOut, totalWrenchOut

def computeCommand(t, q, v, sensor_data, u):
    global com, dcm, zmp, zmp_cmd, totalWrench, qi, dqi, ddqi, t_1, u_1, integral_

    # Get trajectory
    i = int(t * 1.0e3) + 1
    if t > taux :
        p = paux
    else:
        p = 0
    z = zmpRef[i] + p
    c = comRef[i] + p
    vc = dcomRef[i]
    d = c + vc / omega

    # Update state
    com, vcom, dcm, zmp, totalWrench = updateState(robot, q, v, sensor_data)
    comTarget, vcomTarget, dcmTarget, zmpTarget, totalWrenchTarget = \
        updateState(robot, qi, dqi, sensor_data)

    # Update logs (only the value stored by the registered variables using [:])
    dcm_log[:] = dcm
    zmp_log[:] = zmp
    com_log[:] = com
    vcom_log[:] = vcom
    dcmRef_log[axisCom] = d
    zmpRef_log[axisCom] = z
    comRef_log[axisCom] = c
    vcomRef_log[axisCom] = vc
    dcmTarget_log[:] = dcmTarget
    zmpTarget_log[:] = zmpTarget
    comTarget_log[:] = comTarget
    vcomTarget_log[:] = vcomTarget
    totalWrench_angular_log[:] = totalWrench.angular
    totalWrench_linear_log[:] = totalWrench.linear

    # Update targets at HLC frequency
    if int(t * 1.0e3) % int(1.0e3 / fHLC) == 0:
        # Compute zmp command (DCM control)
        if args.targetsFB:
            zi = zmpTarget
            di = dcmTarget
        elif args.mixedFB:
            zi = zmp
            di = com + vcomTarget/omega
        else:
            zi = zmp
            di = dcm

        # KpKdKi dcm
        integral_ = (1 - decay) * integral_ + (t - t_1) * (d - di[axisCom])
        t_1 = t
        zmp_cmd = z - (1 + Kpdcm / omega) * (d - di[axisCom]) \
                + (Kddcm / omega) * (z - zi[axisCom]) - Kidcm / omega * integral_
        if args.clampCmd:
            np.clip(zmp_cmd, -0.1, 0.1, zmp_cmd)

        # Compute joint acceleration from ZMP command
        # Zmp-> com admittance -> com acceleration
        ax = ddcomRef[i] - Acom * (zi[axisCom] - zmp_cmd)
        # Com acceleration -> joint acceleration
        ddqi[iVel] = (ax / (l * np.cos(q[iPos]))) + (v[iVel] ** 2) * np.tan(q[iPos])

        # Compute joint torque from joint acceleration (ID)
        if acceleration_control:
            u_1 = m * (l ** 2) * ((ax / (l * np.cos(q[iPos]))) \
                + (v[iVel] ** 2) * np.tan(q[iPos]) - g * np.sin(q[iPos]) / l)

    # Send last joint torque command
    if acceleration_control:
        u[0] = u_1

    # Integrate last joint acceleration + position control
    elif position_control:
        dqi[iVel] += ddqi[iVel] * 1.0e-3
        qi[iPos] += dqi[iVel] * 1.0e-3
        u[0] = -(Kp * (q[iPos] - qi[iPos]) + Kd * (v[iVel] - dqi[iVel]))

    # Update logs (only the value stored by the registered variables using [:])
    zmp_cmd_log[:] = zmp_cmd
    state_target_log[0], state_target_log[1] = qi[iPos], dqi[iVel]

def internalDynamics(t, q, v, sensor_data, u):
    pass


controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
controller.initialize(robot)
controller.register_variable(["targetPositionPendulum", "targetVelocityPendulum"], state_target_log)
controller.register_variable(["zmpCmdY"], zmp_cmd_log)
controller.register_variable(["zmp" + axis for axis in ["X", "Y"]], zmp_log)
controller.register_variable(["dcm" + axis for axis in SPATIAL_COORDS], dcm_log)
controller.register_variable(["com" + axis for axis in SPATIAL_COORDS], com_log)
controller.register_variable(["vcom" + axis for axis in SPATIAL_COORDS], vcom_log)
controller.register_variable(["zmpTarget" + axis for axis in ["X", "Y"]], zmpTarget_log)
controller.register_variable(["dcmTarget" + axis for axis in SPATIAL_COORDS], dcmTarget_log)
controller.register_variable(["comTarget" + axis for axis in SPATIAL_COORDS], comTarget_log)
controller.register_variable(["vcomTarget" + axis for axis in SPATIAL_COORDS], vcomTarget_log)
controller.register_variable(["wrenchTorque" + axis for axis in SPATIAL_COORDS], totalWrench_angular_log)
controller.register_variable(["wrenchForce" + axis for axis in SPATIAL_COORDS], totalWrench_linear_log)
controller.register_variable(["dcmReference" + axis for axis in SPATIAL_COORDS], dcmRef_log)
controller.register_variable(["comReference" + axis for axis in SPATIAL_COORDS], comRef_log)
controller.register_variable(["vcomReference" + axis for axis in SPATIAL_COORDS], vcomRef_log)
controller.register_variable(["zmpReference" + axis for axis in ["X", "Y"]], zmpRef_log)

# Instantiate the engine
engine = jiminy.Engine()
engine.initialize(robot, controller)

# ######################### Configuration the simulation ################################

robot_options = robot.get_options()
engine_options = engine.get_options()
ctrl_options = controller.get_options()

robot_options["model"]["dynamics"]["enableFlexibleModel"] = False

robot_options["telemetry"]["enableImuSensors"] = True
robot_options["telemetry"]["enableForceSensors"] = True

robot_options["sensors"]['ForceSensor'] = {}
robot_options["sensors"]['ForceSensor']['F1'] = {}
robot_options["sensors"]['ForceSensor']['F1']["noiseStd"] = []
robot_options["sensors"]['ForceSensor']['F1']["bias"] = []
robot_options["sensors"]['ForceSensor']['F1']["delay"] = 0.0
robot_options["sensors"]['ForceSensor']['F1']["delayInterpolationOrder"] = 0
robot_options["sensors"]['ForceSensor']['F2'] = {}
robot_options["sensors"]['ForceSensor']['F2']["noiseStd"] = []
robot_options["sensors"]['ForceSensor']['F2']["bias"] = []
robot_options["sensors"]['ForceSensor']['F2']["delay"] = 0.0
robot_options["sensors"]['ForceSensor']['F2']["delayInterpolationOrder"] = 0
robot_options["sensors"]['ForceSensor']['F3'] = {}
robot_options["sensors"]['ForceSensor']['F3']["noiseStd"] = []
robot_options["sensors"]['ForceSensor']['F3']["bias"] = []
robot_options["sensors"]['ForceSensor']['F3']["delay"] = 0.0
robot_options["sensors"]['ForceSensor']['F3']["delayInterpolationOrder"] = 0
robot_options["sensors"]['ForceSensor']['F4'] = {}
robot_options["sensors"]['ForceSensor']['F4']["noiseStd"] = []
robot_options["sensors"]['ForceSensor']['F4']["bias"] = []
robot_options["sensors"]['ForceSensor']['F4']["delay"] = 0.0
robot_options["sensors"]['ForceSensor']['F4']["delayInterpolationOrder"] = 0

engine_options["telemetry"]["enableConfiguration"] = True
engine_options["telemetry"]["enableVelocity"] = True
engine_options["telemetry"]["enableAcceleration"] = True
engine_options["telemetry"]["enableTorque"] = True
engine_options["telemetry"]["enableEnergy"] = True

engine_options["world"]["gravity"][2] = -9.81
engine_options['world']['groundProfile'] = HeatMapFunctor(0.0, heatMapType_t.CONSTANT) # Force sensor frame offset.

engine_options["stepper"]["solver"] = "runge_kutta_dopri5"  # ["runge_kutta_dopri5", "explicit_euler"]
engine_options["stepper"]["tolRel"] = 1.0e-5
engine_options["stepper"]["tolAbs"] = 1.0e-4
engine_options["stepper"]["dtMax"] = 2.0e-3  # 2.0e-4 for "explicit_euler", 3.0e-3 for "runge_kutta_dopri5"
engine_options["stepper"]["iterMax"] = 100000
engine_options["stepper"]["sensorsUpdatePeriod"] = 1.0e-3
engine_options["stepper"]["controllerUpdatePeriod"] = 1.0e-3
engine_options["stepper"]["logInternalStepperSteps"] = False
engine_options["stepper"]["randomSeed"] = 0

engine_options['contacts']['stiffness'] = 1.0e6
engine_options['contacts']['damping'] = 2000.0 * 2.0
engine_options['contacts']['frictionDry'] = 5.0
engine_options['contacts']['frictionViscous'] = 5.0
engine_options['contacts']['frictionStictionVel'] = 0.01
engine_options['contacts']['frictionStictionRatio'] = 0.5
engine_options['contacts']['transitionEps'] = 0.001

robot.set_options(robot_options)
engine.set_options(engine_options)
controller.set_options(ctrl_options)

# ############################## Run the simulation #####################################

start = time.time()
engine.simulate(tf, x0)
end = time.time()
print("Simulation time: %03.0fms" % ((end - start) * 1.0e3))

# ############################# Extract the results #####################################

log_data, log_constants = engine.get_log()

trajectory_data_log = extract_viewer_data_from_log(log_data, robot)

# Save the log in CSV
engine.write_log(os.path.join(tempfile.gettempdir(), "log.data"), True)

# ############################ Display the results ######################################

if args.plot:
    if args.targetsFB:
        plt.figure("ZMP Y")
        plt.plot(
                 log_data['Global.Time'],
                 log_data['HighLevelController.zmpTargetY'],'b',
                 log_data['Global.Time'],
                 log_data['HighLevelController.zmpCmdY'],'g',
                 log_data['Global.Time'],
                 log_data['HighLevelController.zmpReferenceY'], 'r',
                 log_data['Global.Time'],
                 log_data['HighLevelController.comReferenceY'], 'm')
        plt.legend((
                    "ZMP Y (Targets)",
                    "ZMP CMD Y",
                    "ZMP Reference Y",
                    "COM Reference Y"))
        plt.figure("DCM Y")
        plt.plot(
                 log_data['Global.Time'],
                 log_data['HighLevelController.dcmTargetY'],
                 log_data['Global.Time'],
                 log_data['HighLevelController.dcmReferenceY'])
        plt.legend(("DCM Y (Targets)", "DCM Reference Y"))
    else:
        plt.figure("ZMP Y")
        plt.plot(
                 log_data['Global.Time'],
                 log_data['HighLevelController.zmpY'],'b',
                 log_data['Global.Time'],
                 log_data['HighLevelController.zmpCmdY'],'g',
                 log_data['Global.Time'],
                 log_data['HighLevelController.zmpReferenceY'], 'r',
                 log_data['Global.Time'],
                 log_data['HighLevelController.comReferenceY'], 'm')
        plt.legend((
                    "ZMP Y",
                    "ZMP CMD Y",
                    "ZMP Reference Y",
                    "COM Reference Y"))
        plt.figure("DCM Y")
        plt.plot(
                 log_data['Global.Time'],
                 log_data['HighLevelController.dcmY'],
                 log_data['Global.Time'],
                 log_data['HighLevelController.dcmReferenceY'])
        plt.legend(("DCM Y", "DCM Reference Y"))
    fig = plt.figure("COM Y")
    ax = plt.subplot()
    ax.plot(log_data['Global.Time'],
            log_data['HighLevelController.comY'],
            label = "COM Y")
    ax.plot(log_data['Global.Time'],
            log_data['HighLevelController.comReferenceY'],
            label = "COM Ref Y")
    ax.plot(log_data['Global.Time'],
            log_data['HighLevelController.comTargetY'],
            label = "COM Y (Targets)")
    ax.plot(log_data['Global.Time'],
            log_data['HighLevelController.vcomY'],
            label = "VCOM Y")
    ax.plot(log_data['Global.Time'],
            log_data['HighLevelController.vcomReferenceY'],
            label = "VCOM Ref Y")
    ax.plot(log_data['Global.Time'],
            log_data['HighLevelController.vcomTargetY'],
            label = "VCOM Y (Targets)")
    ax.plot(log_data['Global.Time'],
            log_data['HighLevelController.vcomY']/omega,
            label = "VCOM/omega Y")
    ax.plot(log_data['Global.Time'],
            log_data['HighLevelController.vcomTargetY']/omega,
            label = "VCOM/omega Y (Targets)")
    ax.plot(log_data['Global.Time'],
            log_data['HighLevelController.dcmY'],
            label = "DCM Y")
    ax.plot(log_data['Global.Time'],
            log_data['HighLevelController.dcmReferenceY'],
            label = "DCM Ref Y")
    ax.plot(log_data['Global.Time'],
            log_data['HighLevelController.dcmTargetY'],
            label = "DCM Y (Targets)")
    ax.plot(log_data['Global.Time'],
            log_data['HighLevelController.comTargetY'] + log_data['HighLevelController.vcomTargetY']/omega,
            label = "DCM Y (Mixed)")
    leg = interactive_legend(fig)
    plt.show()

if args.show:
    # Display the simulation trajectory and the reference
    play_trajectories([trajectory_data_log], replay_speed=0.5)
