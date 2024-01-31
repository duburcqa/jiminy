import os

import numpy as np

from jiminy_py.simulator import Simulator

import hppfcl
import pinocchio as pin


# Get script directory
MODULE_DIR = os.path.dirname(__file__)

# Create a gym environment for a simple cube
urdf_path = f"{MODULE_DIR}/../../jiminy_py/unit_py/data/box_collision_mesh.urdf"
simulator = Simulator.build(urdf_path, has_freeflyer=True)

# Sample the initial state
qpos = pin.neutral(simulator.system.robot.pinocchio_model)
qvel = np.zeros(simulator.system.robot.nv)
qpos[2] += 1.5
qvel[0] = 2.0
qvel[3] = 1.0
qvel[5] = 2.0

# Run a simulation
simulator.start(qpos, qvel)

# Create collision detection functor
class CollisionChecker:
    def __init__(self,
                 geom_model: pin.GeometryModel,
                 geom_data: pin.GeometryData,
                 geom_name_1: str,
                 geom_name_2: str) -> None:
        self.geom_name_1 = geom_name_1
        self.geom_name_2 = geom_name_2

        geom_index_1, geom_index_2 = map(
            geom_model.getGeometryId, (geom_name_1, geom_name_2))
        self.oMg1, self.oMg2 = (
            geom_data.oMg[i] for i in (geom_index_1, geom_index_2))
        self.collide_functor = hppfcl.ComputeCollision(*(
            geom_model.geometryObjects[i].geometry
            for i in (geom_index_1, geom_index_2)))
        self.req = hppfcl.CollisionRequest()
        self.req.enable_cached_gjk_guess = True
        self.req.distance_upper_bound = 1e-6
        self.res = hppfcl.CollisionResult()

    def __call__(self) -> bool:
        self.res.clear()
        return bool(self.collide_functor(
            self.oMg1, self.oMg2, self.req, self.res))

if __name__ == '__main__':
    check_collision = CollisionChecker(simulator.robot.collision_model,
                                       simulator.robot.collision_data,
                                       "MassBody_0",
                                       "ground")

    # Run the simulation until collision detection
    while True:
        simulator.step(1e-3)
        if check_collision():
            break
    simulator.stop()

    # Replay the simulation
    simulator.replay(enable_travelling=False, display_contact_frames=True)

