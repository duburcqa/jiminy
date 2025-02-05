# mypy: disable-error-code="no-untyped-def, var-annotated"
""" TODO: Write documentation
"""
import os
import unittest
from typing import Any

import numpy as np

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator

from gym_jiminy.common.envs import BaseJiminyEnv
from gym_jiminy.common.blocks import (
    PDController, PDAdapter, MahonyFilter, DeformationEstimator)
from gym_jiminy.common.bases import ObservedJiminyEnv, ControlledJiminyEnv
from gym_jiminy.common.utils import matrices_to_quat, build_pipeline
from gym_jiminy.envs import AntJiminyEnv


DEBUG = "JIMINY_BUILD_DEBUG" in os.environ


class DeformationEstimatorBlock(unittest.TestCase):
    """ TODO: Write documentation
    """
    def _test_deformation_estimate(self, env, atol):
        # Check that quaternion estimates from MahonyFilter are valid
        true_imu_rots = []
        for imu_sensor in env.robot.sensors['ImuSensor']:
            frame_index = imu_sensor.frame_index
            frame_rot = env.robot.pinocchio_data.oMf[frame_index].rotation
            true_imu_rots.append(frame_rot)
        true_imu_quats = matrices_to_quat(tuple(true_imu_rots))

        features = env.observation['features']
        est_imu_quats = features['mahony_filter']['quat']
        np.testing.assert_allclose(
            true_imu_quats, est_imu_quats, atol=atol)

        # Check that deformation estimates from DeformationEstimator are valid
        model_options = env.robot.get_model_options()
        flexibility_frame_names = [
            flex_options["frameName"]
            for flex_options in model_options["dynamics"]["flexibilityConfig"]]
        est_flex_quats = features['deformation_estimator']['quat']
        est_flex_quats[:] *= np.sign(est_flex_quats[-1])
        for frame_name, joint_index in zip(
                flexibility_frame_names, env.robot.flexibility_joint_indices):
            idx_q = env.robot.pinocchio_model.joints[joint_index].idx_q
            true_flex_quat = env.robot_state.q[idx_q:(idx_q + 4)]
            flex_index = env.observer.flexibility_frame_names.index(frame_name)
            est_flex_quat = est_flex_quats[:, flex_index]
            np.testing.assert_allclose(
                true_flex_quat, est_flex_quat, atol=atol)

    def test_arm(self):
        """ TODO: Write documentation
        """
        import numpy as np

        import jiminy_py.core as jiminy

        # First mount the drive
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        urdf_path = f"{data_dir}/flexible_arm.urdf"
        robot = jiminy.Robot()
        robot.initialize(urdf_path, has_freeflyer=False)

        # Add motor
        motor_joint_name = 'base_to_link1'
        motor = jiminy.SimpleMotor(motor_joint_name)
        motor_options = motor.get_options()
        motor_options["enableVelocityLimit"] = False
        motor.set_options(motor_options)
        robot.attach_motor(motor)
        motor.initialize(motor_joint_name)

        # Add sensors
        encoder = jiminy.EncoderSensor(motor_joint_name)
        robot.attach_sensor(encoder)
        encoder.initialize(motor_joint_name)

        for i in range(1, 5):
            imu = jiminy.ImuSensor(f"link{i + 1}")
            robot.attach_sensor(imu)
            imu.initialize(f"link{i + 1}")

        # We set inertia along non-moving axis to 1.0 for numerical stability
        k_j, d_j = 50.0, 5.0
        model_options = robot.get_model_options()
        model_options['dynamics']['flexibilityConfig'] = [{
            'frameName': f"link{i}_to_link{i+1}",
            'stiffness': k_j * np.ones(3),
            'damping': d_j * np.ones(3),
            'inertia': np.array([1.0, 1.0, 0.0])
        } for i in range(1, 5)
        ]
        robot.set_model_options(model_options)

        # Create a simulator using this robot and controller
        simulator = Simulator(robot)

        # Configure the controller and sensor update periods
        engine_options = simulator.get_options()
        engine_options['stepper']['controllerUpdatePeriod'] = 0.0001
        engine_options['stepper']['sensorsUpdatePeriod'] = 0.0001
        simulator.set_options(engine_options)

        # Instantiate the environment
        env = BaseJiminyEnv(simulator, step_dt=0.01, debug=DEBUG)

        # Add controller and observer blocks
        pd_controller = PDController(
            "pd_controller",
            env,
            kp=150.0,
            kd=0.03,
            update_ratio=1)
        env = ControlledJiminyEnv(env, pd_controller)

        pd_adapter = PDAdapter(
            "pd_adapter",
            env,
            order=1,
            update_ratio=-1)
        env = ControlledJiminyEnv(env, pd_adapter)

        mahony_filter = MahonyFilter(
            "mahony_filter",
            env,
            kp=0.0,
            ki=0.0,
            ignore_twist=False,
            exact_init=True,
            update_ratio=1)
        env = ObservedJiminyEnv(env, mahony_filter)

        deformation_estimator = DeformationEstimator(
            "deformation_estimator",
            env,
            imu_frame_names=tuple(
                sensor.name for sensor in robot.sensors['ImuSensor']),
            flex_frame_names=robot.flexibility_joint_names,
            ignore_twist=True,
            update_ratio=-1)
        env = ObservedJiminyEnv(env, deformation_estimator)

        # Simulate for a while
        env.reset(seed=0)
        for _ in range(100):
            env.step(env.action)

        # Check that deformation estimates matches ground truth
        self._test_deformation_estimate(env, atol=1e-6)

    def test_ant(self):
        """ TODO: Write documentation
        """
        FLEXIBLE_FRAME_NAMES = ("leg_1",
                                "leg_2",
                                "leg_3",
                                "leg_4",
                                "ankle_1",
                                "ankle_2",
                                "ankle_3",
                                "ankle_4")

        IMU_FRAME_NAMES = ("torso",
                           "aux_1",
                           "foot_1",
                           "aux_2",
                           "foot_2",
                           "aux_3",
                           "foot_3",
                           "aux_4",
                           "foot_4")

        # Overload the environment to add deformation points and IMU frames
        class FlexAntJiminyEnv(AntJiminyEnv):
            def __init__(self, debug: bool = False, **kwargs: Any) -> None:
                # Call base implementation
                super().__init__(debug, **kwargs)

                # Add IMU frames
                for frame_name in IMU_FRAME_NAMES:
                    sensor = jiminy.ImuSensor(frame_name)
                    self.robot.attach_sensor(sensor)
                    sensor.initialize(frame_name)

                # Add flexibility frames
                model_options = self.robot.get_model_options()
                model_options["dynamics"]["flexibilityConfig"] = []
                for frame_name in FLEXIBLE_FRAME_NAMES:
                    model_options["dynamics"]["flexibilityConfig"].append(
                        {
                            "frameName": frame_name,
                            "stiffness": np.array([10.0, 10.0, 10.0]),
                            "damping": np.array([0.1, 0.1, 0.1]),
                            "inertia": np.array([0.01, 0.01, 0.01]),
                        }
                    )
                model_options["dynamics"]["enableFlexibility"] = True
                self.robot.set_model_options(model_options)

                # Re-Initialize base class to take into account new sensors
                BaseJiminyEnv.__init__(
                    self,
                    simulator=self.simulator,
                    debug=debug,
                    **{**dict(
                        step_dt=self.step_dt),
                        **kwargs})

                # Configure the controller and sensor update periods
                engine_options = self.simulator.get_options()
                engine_options['stepper']['controllerUpdatePeriod'] = 0.0002
                engine_options['stepper']['sensorsUpdatePeriod'] = 0.0002
                self.simulator.set_options(engine_options)

        # Create pipeline with Mahony filter and DeformationEstimator blocks
        PipelineEnv = build_pipeline(
            env_config=dict(
                cls=FlexAntJiminyEnv
            ),
            layers_config=[
                dict(
                    block=dict(
                        cls=PDController,
                        kwargs=dict(
                            kp=1.0,
                            kd=0.03,
                            update_ratio=1,
                        )
                    )
                ),
                dict(
                    block=dict(
                        cls=PDAdapter,
                        kwargs=dict(
                            order=1,
                            update_ratio=-1,
                        )
                    )
                ),
                dict(
                    block=dict(
                        cls=MahonyFilter,
                        kwargs=dict(
                            kp=0.0,
                            ki=0.0,
                            ignore_twist=False,
                            exact_init=True,
                            update_ratio=1,
                        )
                    )
                ), dict(
                    block=dict(
                        cls=DeformationEstimator,
                        kwargs=dict(
                            imu_frame_names=IMU_FRAME_NAMES,
                            flex_frame_names=FLEXIBLE_FRAME_NAMES,
                            ignore_twist=False,
                            update_ratio=-1,
                        )
                    )
                )
            ]
        )

        # Instantiate the environment
        env = PipelineEnv(debug=DEBUG)

        # Run a simulation
        env.reset(seed=0)
        for _ in range(150):
            env.step(env.action)

        # Check that deformation estimates matches ground truth
        self._test_deformation_estimate(env, atol=1e-4)
