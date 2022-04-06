import os
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt

import jiminy_py.core as jiminy
from jiminy_py.viewer import play_logs_data


# ################################ User parameters #######################################

script_dir = os.path.dirname(os.path.realpath(__file__))
mesh_root_dir = os.path.join(script_dir, "../../data/toys_models")
urdf_path = os.path.join(mesh_root_dir, "double_pendulum/double_pendulum.urdf")

# ########################### Initialize the simulation #################################

# Instantiate the robot
motor_joint_names = ("SecondPendulumJoint",)
robot = jiminy.Robot()
robot.initialize(urdf_path, has_freeflyer=False, mesh_package_dirs=[mesh_root_dir])
for joint_name in motor_joint_names:
    motor = jiminy.SimpleMotor(joint_name)
    robot.attach_motor(motor)
    motor.initialize(joint_name)

# Instantiate the controller
def computeCommand(t, q, v, sensors_data, command):
    pass

def internalDynamics(t, q, v, sensors_data, u_custom):
    pass

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
engine_options["telemetry"]["enableForceExternal"] = False
engine_options["telemetry"]["enableCommand"] = True
engine_options["telemetry"]["enableMotorEffort"] = True
engine_options["telemetry"]["enableEnergy"] = True
engine_options["world"]["gravity"][2] = -9.81
engine_options["stepper"]["solver"] = "runge_kutta_dopri5" # ["runge_kutta_dopri5", "euler_explicit"]
engine_options["stepper"]["tolRel"] = 1.0e-5
engine_options["stepper"]["tolAbs"] = 1.0e-4
engine_options["stepper"]["dtMax"] = 2.0e-3 # 2.0e-4 for "euler_explicit", 3.0e-3 for "runge_kutta_dopri5"
engine_options["stepper"]["iterMax"] = 100000
engine_options["stepper"]["sensorsUpdatePeriod"] = 1.0e-3
engine_options["stepper"]["controllerUpdatePeriod"] = 1.0e-3
engine_options["stepper"]["logInternalStepperSteps"] = False
engine_options["stepper"]["randomSeed"] = 0
engine_options['contacts']['model'] = "spring_damper"
engine_options['contacts']['stiffness'] = 1.0e6
engine_options['contacts']['damping'] = 2000.0
engine_options['contacts']['friction'] = 5.0
engine_options['contacts']['transitionEps'] = 0.001
engine_options['contacts']['transitionVelocity'] = 0.01

robot.set_options(robot_options)
engine.set_options(engine_options)
controller.set_options(ctrl_options)

# ############################## Run the simulation #####################################

q0 = np.zeros((2,))
q0[1] = 0.1
v0 = np.zeros((2,))
tf = 3.0

start = time.time()
engine.simulate(tf, q0, v0)
end = time.time()
print("Simulation time: %03.0fms" %((end - start)*1.0e3))

# ############################# Extract the results #####################################

log_data, log_constants = engine.get_log()
print('%i log points' % log_data['Global.Time'].shape)
print(log_constants)

# Save the log in HDF5
engine.write_log(os.path.join(tempfile.gettempdir(), "log.hdf5"), format="hdf5")

# ############################ Display the results ######################################

# Plot some data using standard tools only
plt.figure()
plt.plot(log_data['Global.Time'], log_data['HighLevelController.energy'])
plt.title('Double pendulum energy')
plt.grid()
plt.show()

# Display the simulation trajectory and the reference
play_logs_data(robot, log_data,
               speed_ratio=0.5,
               camera_xyzrpy=[(0.0, 7.0, 0.0), (np.pi/2, 0.0, np.pi)],
               delete_robot_on_close=False)
