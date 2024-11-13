""" TODO: Write documentation
"""
from operator import sub
import unittest

import numpy as np

import gymnasium as gym
from jiminy_py.log import extract_trajectory_from_log

from gym_jiminy.common.utils import (
    quat_difference, matrix_to_quat, matrix_to_rpy)
from gym_jiminy.common.bases import EpisodeState, ComposedJiminyEnv
from gym_jiminy.common.quantities import (
    OrientationType, FramePosition, FrameOrientation)
from gym_jiminy.common.compositions import (
    DriftTrackingQuantityTermination,
    ShiftTrackingQuantityTermination,
    BaseRollPitchTermination,
    FallingTermination,
    FootCollisionTermination,
    MechanicalSafetyTermination,
    FlyingTermination,
    ImpactForceTermination,
    MechanicalPowerConsumptionTermination,
    DriftTrackingBaseOdometryPositionTermination,
    DriftTrackingBaseOdometryOrientationTermination,
    ShiftTrackingMotorPositionsTermination,
    ShiftTrackingFootOdometryPositionsTermination,
    ShiftTrackingFootOdometryOrientationsTermination)


class TerminationConditions(unittest.TestCase):
    """ TODO: Write documentation
    """
    def setUp(self):
        self.env = gym.make("gym_jiminy.envs:atlas-pid", debug=False)

        self.env.eval()
        self.env.reset(seed=1)
        action = 0.5 * self.env.action_space.sample()
        for _ in range(25):
            self.env.step(action)
        self.env.stop()
        trajectory = extract_trajectory_from_log(self.env.log_data)
        self.env.train()

        self.env.quantities.add_trajectory("reference", trajectory)
        self.env.quantities.select_trajectory("reference")

    def test_composition(self):
        """ TODO: Write documentation
        """
        ROLL_MIN, ROLL_MAX = -0.2, 0.2
        PITCH_MIN, PITCH_MAX = -0.05, 0.3
        termination = BaseRollPitchTermination(
            self.env,
            np.array([ROLL_MIN, PITCH_MIN]),
            np.array([ROLL_MAX, PITCH_MAX]))
        self.env.reset(seed=0)
        env = ComposedJiminyEnv(self.env, terminations=(termination,))

        env.reset(seed=0)
        action = self.env.action_space.sample()
        for _ in range(20):
            _, _, terminated_env, _, _ = env.step(action)
            terminated_cond, _ = termination({})
            assert not (terminated_env ^ terminated_cond)
            if terminated_env:
                terminated_unwrapped, _ = env.unwrapped.has_terminated({})
                assert not terminated_unwrapped
                break

    def test_drift_tracking(self):
        """ TODO: Write documentation
        """
        termination_pos_config = ("pos", (FramePosition, {}), -0.2, 0.3, sub)
        termination_rot_config = (
            "rot",
            (FrameOrientation, dict(type=OrientationType.QUATERNION)),
            np.array([-0.5, -0.6, -0.7]),
            np.array([0.7, 0.5, 0.6]),
            quat_difference)

        for i, (is_truncation, is_training_only) in enumerate((
            (False, False), (True, False), (False, True))):
            termination_pos, termination_rot = (
                DriftTrackingQuantityTermination(
                    self.env,
                    f"drift_tracking_{name}_{i}",
                    lambda mode: (quantity_cls, dict(
                        **quantity_kwargs,
                        frame_name="root_joint",
                        mode=mode)),
                    low=low,
                    high=high,
                    horizon=0.3,
                    grace_period=0.2,
                    op=op,
                    is_truncation=is_truncation,
                    is_training_only=is_training_only
                ) for name, (quantity_cls, quantity_kwargs), low, high, op in (
                    termination_pos_config, termination_rot_config))

            self.env.stop()
            self.env.eval()
            self.env.reset(seed=0)
            action = self.env.action_space.sample()
            oMf = self.env.robot.pinocchio_data.oMf[1]
            position, rotation = oMf.translation, oMf.rotation

            positions, rotations = [], []
            for _ in range(25):
                info_pos, info_rot = {}, {}
                flags_pos = termination_pos(info_pos)
                flags_rot = termination_rot(info_rot)

                positions.append(position.copy())
                rotations.append(matrix_to_quat(rotation))

                for termination, (terminated, truncated), values, info in (
                        (termination_pos, flags_pos, positions, info_pos),
                        (termination_rot, flags_rot, rotations, info_rot)):
                    values = values[-termination.max_stack:]
                    drift = termination.op(values[-1], values[0])
                    value = termination.data.quantity_left.get()
                    np.testing.assert_allclose(drift, value)

                    time = self.env.stepper_state.t
                    is_active = (
                        time >= termination.grace_period and
                        not termination.is_training_only)
                    assert info == {
                        termination.name: EpisodeState.TERMINATED
                        if terminated else EpisodeState.TRUNCATED
                        if truncated else EpisodeState.CONTINUED}
                    if terminated or truncated:
                        assert is_active
                        assert terminated ^ termination.is_truncation
                    elif is_active:
                        value = termination.data.get()
                        assert np.all(value >= termination.low)
                        assert np.all(value <= termination.high)
                _, _, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    break

    def test_shift_tracking(self):
        """ TODO: Write documentation
        """
        termination_pos_config = ("pos", (FramePosition, {}), 0.1, sub)
        termination_rot_config = (
            "rot",
            (FrameOrientation, dict(type=OrientationType.QUATERNION)),
            0.3,
            quat_difference)

        for i, (is_truncation, is_training_only) in enumerate((
            (False, False), (True, False), (False, True))):
            termination_pos, termination_rot = (
                ShiftTrackingQuantityTermination(
                    self.env,
                    f"shift_tracking_{name}_{i}",
                    lambda mode: (quantity_cls, dict(
                        **quantity_kwargs,
                        frame_name="root_joint",
                        mode=mode)),
                    thr=thr,
                    horizon=0.3,
                    grace_period=0.2,
                    op=op,
                    is_truncation=is_truncation,
                    is_training_only=is_training_only
                ) for name, (quantity_cls, quantity_kwargs), thr, op in (
                    termination_pos_config, termination_rot_config))

            self.env.stop()
            self.env.eval()
            self.env.reset(seed=0)
            action = self.env.action_space.sample()
            oMf = self.env.robot.pinocchio_data.oMf[1]
            position, rotation = oMf.translation, oMf.rotation

            positions, rotations = [], []
            for _ in range(25):
                info_pos, info_rot = {}, {}
                flags_pos = termination_pos(info_pos)
                flags_rot = termination_rot(info_rot)

                positions.append(position.copy())
                rotations.append(matrix_to_quat(rotation))

                for termination, (terminated, truncated), values, info in (
                        (termination_pos, flags_pos, positions, info_pos),
                        (termination_rot, flags_rot, rotations, info_rot)):
                    values = values[-termination.max_stack:]
                    stack = np.stack(values, axis=-1)
                    left = termination.data.quantity_left.get()
                    np.testing.assert_allclose(stack, left)
                    right = termination.data.quantity_right.get()
                    diff = termination.op(left, right)
                    shift = np.min(np.linalg.norm(
                        diff.reshape((-1, len(values))), axis=0))
                    value = termination.data.get()
                    np.testing.assert_allclose(shift, value)

                    time = self.env.stepper_state.t
                    is_active = (
                        time >= termination.grace_period and
                        not termination.is_training_only)
                    assert info == {
                        termination.name: EpisodeState.TERMINATED
                        if terminated else EpisodeState.TRUNCATED
                        if truncated else EpisodeState.CONTINUED}
                    if terminated or truncated:
                        assert is_active
                        assert terminated ^ termination.is_truncation
                    elif is_active:
                        assert np.all(value <= termination.high)
                _, _, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    break

    def test_base_roll_pitch(self):
        """ TODO: Write documentation
        """
        ROLL_MIN, ROLL_MAX = -0.2, 0.2
        PITCH_MIN, PITCH_MAX = -0.05, 0.3
        roll_pitch_termination = BaseRollPitchTermination(
            self.env,
            np.array([ROLL_MIN, PITCH_MIN]),
            np.array([ROLL_MAX, PITCH_MAX]))

        self.env.reset(seed=0)
        rotation = self.env.robot.pinocchio_data.oMf[1].rotation
        action = self.env.action_space.sample()
        for _ in range(20):
            _, _, terminated, _, _ = self.env.step(action)
            if terminated:
                break
            terminated, truncated = roll_pitch_termination({})
            assert not truncated
            roll, pitch, _ = matrix_to_rpy(rotation)
            is_valid = (
                ROLL_MIN < roll < ROLL_MAX and PITCH_MIN < pitch < PITCH_MAX)
            assert terminated ^ is_valid

    def test_foot_collision(self):
        """ TODO: Write documentation
        """
        termination = FootCollisionTermination(self.env, security_margin=0.0)

        motor_names = [motor.name for motor in self.env.robot.motors]
        left_motor_index = motor_names.index('l_leg_hpx')
        right_motor_index = motor_names.index('r_leg_hpx')
        action = np.zeros((len(motor_names),))
        action[[left_motor_index, right_motor_index]] = -0.5, 0.5

        self.env.robot.remove_contact_points([])
        self.env.stop()
        self.env.eval()
        self.env.reset(seed=0)
        for _ in range(10):
            self.env.step(action)
            terminated, truncated = termination({})
            assert not truncated
            if terminated:
                break
        else:
            raise AssertionError("No collision detected.")

    def test_safety_limits(self):
        """ TODO: Write documentation
        """
        POSITION_MARGIN, VELOCITY_MAX = 0.05, 1.0
        termination = MechanicalSafetyTermination(
            self.env, POSITION_MARGIN, VELOCITY_MAX)

        self.env.reset(seed=0)

        position_indices, velocity_indices = [], []
        pincocchio_model = self.env.robot.pinocchio_model
        for motor in self.env.robot.motors:
            joint = pincocchio_model.joints[motor.joint_index]
            position_indices.append(joint.idx_q)
            velocity_indices.append(joint.idx_v)
        position_lower = pincocchio_model.lowerPositionLimit[position_indices]
        position_lower += POSITION_MARGIN
        position_upper = pincocchio_model.upperPositionLimit[position_indices]
        position_upper -= POSITION_MARGIN

        action = self.env.action_space.sample()
        for _ in range(20):
            _, _, terminated, _, _ = self.env.step(action)
            if terminated:
                break
            terminated, truncated = termination({})
            position = self.env.robot_state.q[position_indices]
            velocity = self.env.robot_state.v[velocity_indices]
            is_valid = np.all(
                (position_lower <= position) | (velocity >= - VELOCITY_MAX))
            is_valid = is_valid and np.all(
                (position_upper >= position) | (velocity <= VELOCITY_MAX))
            assert terminated ^ is_valid

    def test_flying(self):
        """ TODO: Write documentation
        """
        MAX_HEIGHT = 0.02
        termination = FlyingTermination(self.env, max_height=MAX_HEIGHT)

        self.env.reset(seed=0)

        engine_options = self.env.unwrapped.engine.get_options()
        heightmap = engine_options["world"]["groundProfile"]

        action = self.env.action_space.sample()
        for _ in range(20):
            _, _, terminated, _, _ = self.env.step(action)
            if terminated:
                break
            terminated, truncated = termination({})
            is_valid = False
            for frame_index in self.env.robot.contact_frame_indices:
                transform = self.env.robot.pinocchio_data.oMf[frame_index]
                position = transform.translation
                height, normal = heightmap(position[:2])
                depth = (position[2] - height) * normal[2]
                if depth <= MAX_HEIGHT:
                    is_valid = True
                    break
            assert terminated ^ is_valid

    def test_drift_tracking_base_odom(self):
        """ TODO: Write documentation
        """
        MAX_POS_ERROR, MAX_ROT_ERROR = 0.1, 0.2
        termination_pos = DriftTrackingBaseOdometryPositionTermination(
            self.env, MAX_POS_ERROR, 1.0)
        quantity_pos = termination_pos.data
        termination_rot = DriftTrackingBaseOdometryOrientationTermination(
            self.env, MAX_ROT_ERROR, 1.0)
        quantity_rot = termination_rot.data

        self.env.reset(seed=0)
        action = self.env.action_space.sample()
        for _ in range(20):
            _, _, terminated, _, _ = self.env.step(action)
            if terminated:
                break
            terminated, truncated = termination_pos({})
            value_left = quantity_pos.quantity_left.get()
            value_right = quantity_pos.quantity_right.get()
            diff = value_left - value_right
            is_valid = np.linalg.norm(diff) <= MAX_POS_ERROR
            assert terminated ^ is_valid
            value_left = quantity_rot.quantity_left.get()
            value_right = quantity_rot.quantity_right.get()
            diff = value_left - value_right
            terminated, truncated = termination_rot({})
            is_valid = np.abs(diff) <= MAX_ROT_ERROR
            assert terminated ^ is_valid

    def test_misc(self):
        """ TODO: Write documentation
        """
        for termination in (
                FallingTermination(self.env, 0.6),
                ImpactForceTermination(self.env, 1.0),
                MechanicalPowerConsumptionTermination(self.env, 400.0, 1.0),
                ShiftTrackingMotorPositionsTermination(self.env, 0.4, 0.5),
                ShiftTrackingFootOdometryPositionsTermination(
                    self.env, 0.2, 0.5),
                ShiftTrackingFootOdometryOrientationsTermination(
                    self.env, 0.1, 0.5)):
            self.env.stop()
            self.env.eval()
            self.env.reset(seed=0)
            action = self.env.action_space.sample()
            for _ in range(20):
                _, _, terminated, _, _ = self.env.step(action)
                terminated, truncated = termination({})
                assert not truncated
