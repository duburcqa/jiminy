import os
import time
import numpy as np

import jiminy
from jiminy_py.engine_asynchronous import EngineAsynchronous


os.environ["JIMINY_MESH_PATH"] = os.path.join(os.environ["HOME"], "wdc_workspace/src/jiminy/data")
urdf_path = os.path.join(os.environ["JIMINY_MESH_PATH"], "cartpole/cartpole.urdf")

contacts = []
motors = ["slider_to_cart"]
model = jiminy.Model()
model.initialize(urdf_path, contacts, motors, False)
model.add_encoder_sensor("slider", "slider_to_cart")
model.add_encoder_sensor("pole", "cart_to_pole")
engine_py = EngineAsynchronous(model)

model_options = model.get_model_options()
sensors_options = model.get_sensors_options()
engine_options = engine_py.get_engine_options()
ctrl_options = engine_py.get_controller_options()

model_options["telemetry"]["enableImuSensors"] = False
engine_options["telemetry"]["enableConfiguration"] = False
engine_options["telemetry"]["enableVelocity"] = False
engine_options["telemetry"]["enableAcceleration"] = False
engine_options["telemetry"]["enableCommand"] = False
engine_options["telemetry"]["enableEnergy"] = False

engine_options["stepper"]["solver"] = "runge_kutta_dopri5"
engine_options["stepper"]["iterMax"] = -1
engine_options["stepper"]["sensorsUpdatePeriod"] = 1e-3
engine_options["stepper"]["controllerUpdatePeriod"] = 1e-3

model.set_model_options(model_options)
model.set_sensors_options(sensors_options)
engine_py.set_engine_options(engine_options)
engine_py.set_controller_options(ctrl_options)

time_start = time.time()
engine_py.seed(0)
engine_py.reset()
for i in range(10000):
    engine_py.step(np.array([0.001]))
    engine_py.render()

    time_end = time.time()
    dt = engine_options["stepper"]["controllerUpdatePeriod"] - (time_end - time_start)
    time.sleep(max(dt, 0.0))
    time_start = time_end
