import os
import time
import numpy as np

from jiminy_py import core as jiminy
from jiminy_py.engine_asynchronous import EngineAsynchronous


os.environ["JIMINY_MESH_PATH"] = os.path.join(os.environ["HOME"], "wdc_workspace/src/jiminy/data")
urdf_path = os.path.join(os.environ["JIMINY_MESH_PATH"], "cartpole/cartpole.urdf")

motor_joint_names = ("slider_to_cart",)
encoder_sensors_def = {"slider": "slider_to_cart", "pole": "cart_to_pole"}

robot = jiminy.Robot()
robot.initialize(urdf_path, False)
for joint_name in motor_joint_names:
    motor = jiminy.SimpleMotor(joint_name)
    robot.attach_motor(motor)
    motor.initialize(joint_name)
for sensor_name, joint_name in encoder_sensors_def.items():
    encoder = jiminy.EncoderSensor(sensor_name)
    robot.attach_sensor(encoder)
    encoder.initialize(joint_name)

engine_py = EngineAsynchronous(robot)

robot_options = robot.get_options()
engine_options = engine_py.get_engine_options()
ctrl_options = engine_py.get_controller_options()

robot_options["telemetry"]["enableImuSensors"] = False
engine_options["telemetry"]["enableConfiguration"] = False
engine_options["telemetry"]["enableVelocity"] = False
engine_options["telemetry"]["enableAcceleration"] = False
engine_options["telemetry"]["enableTorque"] = False
engine_options["telemetry"]["enableEnergy"] = False

engine_options["stepper"]["solver"] = "runge_kutta_dopri5"
engine_options["stepper"]["iterMax"] = -1
engine_options["stepper"]["sensorsUpdatePeriod"] = 1e-3
engine_options["stepper"]["controllerUpdatePeriod"] = 1e-3

robot.set_options(robot_options)
engine_py.set_engine_options(engine_options)
engine_py.set_controller_options(ctrl_options)

time_start = time.time()
engine_py.seed(0)
engine_py.reset(np.zeros(robot.nx))
for i in range(10000):
    engine_py.step(np.array([0.001]))
    engine_py.render()

    time_end = time.time()
    dt = engine_options["stepper"]["controllerUpdatePeriod"] - (time_end - time_start)
    time.sleep(max(dt, 0.0))
    time_start = time_end
