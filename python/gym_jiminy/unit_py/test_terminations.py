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
    BaseRollPitchTermination)


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
                    max_stack=8,
                    grace_period=0.2,
                    op=op,
                    is_truncation=is_truncation,
                    is_training_only=is_training_only
                ) for name, (quantity_cls, quantity_kwargs), low, high, op in (
                    termination_pos_config, termination_rot_config))

            self.env.reset(seed=0)
            self.env.eval()
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
                    data = termination.quantity.value_left
                    np.testing.assert_allclose(drift, data)

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
                        value = termination.quantity.get()
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
                    max_stack=8,
                    grace_period=0.2,
                    op=op,
                    is_truncation=is_truncation,
                    is_training_only=is_training_only
                ) for name, (quantity_cls, quantity_kwargs), thr, op in (
                    termination_pos_config, termination_rot_config))

            self.env.reset(seed=0)
            self.env.eval()
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
                    left = termination.quantity.value_left
                    np.testing.assert_allclose(stack, left)
                    right = termination.quantity.value_right
                    diff = termination.op(left, right)
                    shift = np.min(np.linalg.norm(
                        diff.reshape((-1, len(values))), axis=0))
                    value = termination.quantity.get()
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
