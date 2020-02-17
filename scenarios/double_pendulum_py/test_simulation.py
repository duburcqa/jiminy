import os
import time
import numpy as np
import matplotlib.pyplot as plt

import jiminy
from jiminy_py.viewer import play_trajectories
from jiminy_py.log import extract_state_from_simulation_log

# ################################ User parameters #######################################

os.environ["JIMINY_MESH_PATH"] = os.path.join(os.environ["HOME"], "wdc_workspace/src/jiminy/data")
urdf_path = os.path.join(os.environ["JIMINY_MESH_PATH"], "double_pendulum/double_pendulum.urdf")

# ########################### Initialize the simulation #################################

# Instantiate the model
contacts = []
motors = ["SecondPendulumJoint"]
model = jiminy.Model()
model.initialize(urdf_path, contacts, motors, False)

# Instantiate the controller
def computeCommand(t, q, v, sensor_data, u):
    u[0] = 0.0

def internalDynamics(t, q, v, sensor_data, u):
    u[:] = 0.0

controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
controller.initialize(model)

# Instantiate the engine
engine = jiminy.Engine()
engine.initialize(model, controller)

# ######################### Configuration the simulation ################################

model_options = model.get_model_options()
sensors_options = model.get_sensors_options()
engine_options = engine.get_options()
ctrl_options = controller.get_options()

model_options["telemetry"]["enableImuSensors"] = True
engine_options["telemetry"]["enableConfiguration"] = True
engine_options["telemetry"]["enableVelocity"] = True
engine_options["telemetry"]["enableAcceleration"] = True
engine_options["telemetry"]["enableCommand"] = True
engine_options["telemetry"]["enableEnergy"] = True
engine_options["world"]["gravity"][2] = -9.81
engine_options["stepper"]["solver"] = "runge_kutta_dopri5" # ["runge_kutta_dopri5", "explicit_euler"]
engine_options["stepper"]["tolRel"] = 1.0e-5
engine_options["stepper"]["tolAbs"] = 1.0e-4
engine_options["stepper"]["dtMax"] = 2.0e-3 # 2.0e-4 for "explicit_euler", 3.0e-3 for "runge_kutta_dopri5"
engine_options["stepper"]["iterMax"] = 100000
engine_options["stepper"]["sensorsUpdatePeriod"] = 1.0e-3
engine_options["stepper"]["controllerUpdatePeriod"] = 1.0e-3
engine_options["stepper"]["randomSeed"] = 0
engine_options['contacts']['stiffness'] = 1.0e6
engine_options['contacts']['damping'] = 2000.0
engine_options['contacts']['dryFrictionVelEps'] = 0.01
engine_options['contacts']['frictionDry'] = 5.0
engine_options['contacts']['frictionViscous'] = 5.0
engine_options['contacts']['transitionEps'] = 0.001

model.set_model_options(model_options)
model.set_sensors_options(sensors_options)
engine.set_options(engine_options)
controller.set_options(ctrl_options)

# ############################## Run the simulation #####################################

x0 = np.zeros((4,))
x0[1] = 0.1
tf = 3.0

start = time.time()
engine.simulate(x0, tf)
end = time.time()
print("Simulation time: %03.0fms" %((end - start)*1.0e3))

# ############################# Extract the results #####################################

log_data, log_constants = engine.get_log()
print('%i log points' % log_data['Global.Time'].shape)
print(log_constants)
trajectory_data_log = extract_state_from_simulation_log(log_data, model)

# Save the log in CSV
engine.write_log("/tmp/log.data", False)

# ############################ Display the results ######################################

# Plot some data using standard tools only
plt.figure()
plt.plot(log_data['Global.Time'], log_data['HighLevelController.energy'])
plt.title('Double pendulum energy')
plt.grid()
plt.show()

# Display the simulation trajectory and the reference
play_trajectories([trajectory_data_log], speed_ratio=0.5)
