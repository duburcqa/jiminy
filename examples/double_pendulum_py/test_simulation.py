import os
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt

from jiminy_py import core as jiminy
from jiminy_py.viewer import extract_viewer_data_from_log, play_trajectories


# ################################ User parameters #######################################

script_dir = os.path.dirname(os.path.realpath(__file__))
os.environ["JIMINY_MESH_PATH"] = os.path.join(script_dir, "../../data")
urdf_path = os.path.join(os.environ["JIMINY_MESH_PATH"], "double_pendulum/double_pendulum.urdf")

# ########################### Initialize the simulation #################################

# Instantiate the robot
motor_joint_names = ("SecondPendulumJoint",)
robot = jiminy.Robot()
robot.initialize(urdf_path, False)
for joint_name in motor_joint_names:
    motor = jiminy.SimpleMotor(joint_name)
    robot.attach_motor(motor)
    motor.initialize(joint_name)

# Instantiate the controller
def computeCommand(t, q, v, sensor_data, u):
    u[0] = 0.0

def internalDynamics(t, q, v, sensor_data, u):
    u[:] = 0.0

controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
controller.initialize(robot)

# Instantiate the engine
engine = jiminy.Engine()
engine.initialize(robot, controller)

# ######################### Configuration the simulation ################################

robot_options = robot.get_options()
engine_options = engine.get_options()
ctrl_options = controller.get_options()

robot_options["telemetry"]["enableImuSensors"] = True
engine_options["telemetry"]["enableConfiguration"] = True
engine_options["telemetry"]["enableVelocity"] = True
engine_options["telemetry"]["enableAcceleration"] = True
engine_options["telemetry"]["enableTorque"] = True
engine_options["telemetry"]["enableEnergy"] = True
engine_options["world"]["gravity"][2] = -9.81
engine_options["stepper"]["solver"] = "runge_kutta_dopri5" # ["runge_kutta_dopri5", "explicit_euler"]
engine_options["stepper"]["tolRel"] = 1.0e-5
engine_options["stepper"]["tolAbs"] = 1.0e-4
engine_options["stepper"]["dtMax"] = 2.0e-3 # 2.0e-4 for "explicit_euler", 3.0e-3 for "runge_kutta_dopri5"
engine_options["stepper"]["iterMax"] = 100000
engine_options["stepper"]["sensorsUpdatePeriod"] = 1.0e-3
engine_options["stepper"]["controllerUpdatePeriod"] = 1.0e-3
engine_options["stepper"]["logInternalStepperSteps"] = False
engine_options["stepper"]["randomSeed"] = 0
engine_options['contacts']['stiffness'] = 1.0e6
engine_options['contacts']['damping'] = 2000.0
engine_options['contacts']['frictionDry'] = 5.0
engine_options['contacts']['frictionViscous'] = 5.0
engine_options['contacts']['frictionStictionVel'] = 0.01
engine_options['contacts']['frictionStictionRatio'] = 0.5
engine_options['contacts']['transitionEps'] = 0.001

robot.set_options(robot_options)
engine.set_options(engine_options)
controller.set_options(ctrl_options)

# ############################## Run the simulation #####################################

x0 = np.zeros((4,))
x0[1] = 0.1
tf = 3.0

start = time.time()
engine.simulate(tf, x0)
end = time.time()
print("Simulation time: %03.0fms" %((end - start)*1.0e3))

# ############################# Extract the results #####################################

log_data, log_constants = engine.get_log()
print('%i log points' % log_data['Global.Time'].shape)
print(log_constants)
trajectory_data_log = extract_viewer_data_from_log(log_data, robot)

# Save the log in CSV
engine.write_log(os.path.join(tempfile.gettempdir(), "log.csv"), False)

# ############################ Display the results ######################################

# Plot some data using standard tools only
plt.figure()
plt.plot(log_data['Global.Time'], log_data['HighLevelController.energy'])
plt.title('Double pendulum energy')
plt.grid()
plt.show()

# Display the simulation trajectory and the reference
play_trajectories([trajectory_data_log], replay_speed=0.5,
                  camera_xyzrpy=[0.0, 7.0, 0.0, np.pi/2, 0.0, np.pi])
