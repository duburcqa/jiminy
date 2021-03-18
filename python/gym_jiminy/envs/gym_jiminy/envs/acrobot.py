import os
import numpy as np
from pkg_resources import resource_filename
from typing import Optional, Tuple, Dict, Any

import gym

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator

from gym_jiminy.common.utils import sample, SpaceDictNested
from gym_jiminy.common.envs import BaseJiminyEnv, BaseJiminyGoalEnv


# Stepper update period
STEP_DT = 0.2
# Range of uniform sampling distribution of joint angles
THETA_RANDOM_MAX = 0.1
# Range of uniform sampling distribution of joint velocities
DTHETA_RANDOM_MAX = 0.1
# Relative height of tip to consider an episode successful for normal env
HEIGHT_REL_DEFAULT_THRESHOLD = 0.5
# Range of rel. height of tip to consider an episode successful for goal env
HEIGHT_REL_GOAL_THRESHOLD_RANGE = (-0.2, 0.98)
# Standard deviation of the noise added to the action
ACTION_NOISE = 0.0


class AcrobotJiminyEnv(BaseJiminyEnv):
    """Implementation of a Gym environment for the Acrobot which is using
    Jiminy Engine to perform physics computations and Meshcat for rendering.

    It is a specialization of BaseJiminyEnv. The acrobot is a 2-link
    pendulum with only the second joint actuated. Initially, both links point
    downwards. The goal is to swing the end-effector at a height at least the
    length of one link above the base. Both links can swing freely and can pass
    by each other, i.e. they don't collide when they have the same angle.

    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :

        [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]

    For the first link, an angle of 0 corresponds to the link pointing
    downwards. The angle of the second link is relative to the angle of the
    first link. An angle of 0 corresponds to having the same angle between the
    two links. A state of [1, 0, 1, 0, ..., ...] means that both links point
    downwards.

    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between the
    two pendulum links.

    .. note::
        - R. Sutton: Generalization in Reinforcement Learning:
             Successful Examples Using Sparse Coarse Coding (NIPS 1996)
        - R. Sutton and A. G. Barto:
             Reinforcement learning: An introduction.
             Cambridge: MIT press, 1998.
    """
    def __init__(self, continuous: bool = False, debug: bool = False) -> None:
        """
        :param continuous: Whether or not the action space is continuous. If
                           not continuous, the action space has only 3 states,
                           i.e. low, zero, and high.
                           Optional: True by default.
        """
        # Backup some input arguments
        self.continuous = continuous

        # Get URDF path
        data_dir = resource_filename(
            "gym_jiminy.envs", "data/toys_models/acrobot")
        urdf_path = os.path.join(
            data_dir, "acrobot.urdf")

        # Instantiate robot
        robot = jiminy.Robot()
        robot.initialize(
            urdf_path, has_freeflyer=False, mesh_package_dirs=[data_dir])

        # Add motors and sensors
        motor_joint_name = "SecondArmJoint"
        encoder_joint_names = ("FirstArmJoint", "SecondArmJoint")
        motor = jiminy.SimpleMotor(motor_joint_name)
        robot.attach_motor(motor)
        motor.initialize(motor_joint_name)
        for joint_name in encoder_joint_names:
            encoder = jiminy.EncoderSensor(joint_name)
            robot.attach_sensor(encoder)
            encoder.initialize(joint_name)

        # Instantiate simulator
        simulator = Simulator(robot)

        # Map between discrete actions and actual motor torque if necessary
        if not self.continuous:
            self.AVAIL_CTRL = [-motor.command_limit, 0.0, motor.command_limit]

        # Internal parameters used for computing termination condition
        self._tipIdx = robot.pinocchio_model.getFrameId("Tip")
        self._tipPosZMax = abs(robot.pinocchio_data.oMf[
            self._tipIdx].translation[[2]])

        # Configure the learning environment
        super().__init__(simulator, step_dt=STEP_DT, debug=debug)

        # Create some proxies for fast access
        self.__state_view = (self._observation[:self.robot.nq],
                             self._observation[-self.robot.nv:])

    def _setup(self) -> None:
        """ TODO: Write documentation.
        """
        # Call base implementation
        super()._setup()

        # Increase stepper accuracy for time-continuous control
        engine_options = self.simulator.engine.get_options()
        engine_options["stepper"]["solver"] = "runge_kutta_4"
        engine_options["stepper"]["dtMax"] = min(STEP_DT, 0.02)
        self.simulator.engine.set_options(engine_options)

    def _refresh_observation_space(self) -> None:
        """Configure the observation of the environment.

        Only the state is observable, while by default, the current time,
        state, and sensors data are available.
        """
        # TODO: `gym.spaces.flatten_space` does not properly handle dtype
        # before gym>=0.18.0, which is not compatible with Python 3.9...
        state_subspaces = self._get_state_space().spaces.values()
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([s.low for s in state_subspaces]),
            high=np.concatenate([s.high for s in state_subspaces]),
            dtype=np.result_type(*[s.dtype for s in state_subspaces]))

    def refresh_observation(self, *args: Any, **kwargs: Any) -> None:
        """Update the observation based on the current simulation state.

        Only the state is observable, while by default, the current time,
        state, and sensors data are available.

        .. note::
            For goal env, in addition of the current robot state, both the
            desired and achieved goals are observable.
        """
        if not self.simulator.is_simulation_running:
            self.__state = (self.system_state.q, self.system_state.v)
        self.__state_view[0][:] = self.__state[0]
        self.__state_view[1][:] = self.__state[1]

    def _refresh_action_space(self) -> None:
        """Configure the action space of the environment.

        Replace the action space by its discrete representation depending on
        'continuous'.
        """
        if not self.continuous:
            self.action_space = gym.spaces.Discrete(len(self.AVAIL_CTRL))
        else:
            super()._refresh_action_space()

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a valid configuration and velocity for the robot.

        The initial state is randomly sampled using a hypercube uniform
        distribution, according to the official Gym acrobot-v1 action space.

        See documentation: https://gym.openai.com/envs/Acrobot-v1/.
        """
        theta1, theta2 = sample(
            scale=THETA_RANDOM_MAX, shape=(2,), rg=self.rg)
        qpos = np.array([np.cos(theta1), np.sin(theta1),
                         np.cos(theta2), np.sin(theta2)])
        qvel = sample(
            scale=DTHETA_RANDOM_MAX, shape=(2,), rg=self.rg)
        return qpos, qvel

    def _get_achieved_goal(self) -> np.ndarray:
        """Compute achieved goal based on current state of the robot.

        It corresponds to the position of the tip of the acrobot.
        """
        tip_transform = self.robot.pinocchio_data.oMf[self._tipIdx]
        tip_position_z = tip_transform.translation[[2]]
        return tip_position_z

    def is_done(self) -> bool:  # type: ignore[override]
        """Determine whether a termination condition has been reached.

        The episode terminates if the goal has been achieved, namely if the tip
        of the acrobot is above 'HEIGHT_REL_DEFAULT_THRESHOLD'.
        """
        # pylint: disable=arguments-differ

        achieved_goal = self._get_achieved_goal()
        desired_goal = HEIGHT_REL_DEFAULT_THRESHOLD * self._tipPosZMax
        return bool(achieved_goal > desired_goal)

    def compute_command(self,
                        measure: SpaceDictNested,
                        action: np.ndarray
                        ) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        Convert a discrete action into its actual value if necessary, then add
        noise to the action is enable.

        :param measure: Observation of the environment.
        :param action: Desired motors efforts.
        """
        # Call base implementation
        action = super().compute_command(measure, action)

        # Compute the actual torque to apply
        if not self.continuous:
            action = self.AVAIL_CTRL[action]
        if ACTION_NOISE > 0.0:
            action += sample(scale=ACTION_NOISE, rg=self.rg)

        return action

    def compute_reward(self,  # type: ignore[override]
                       info: Dict[str, Any]) -> float:
        """Compute reward at current episode state.

        Get a small negative reward till success.
        """
        # pylint: disable=arguments-differ

        if self.is_done():
            reward = 0.0
        else:
            reward = -1.0
        return reward

    def render(self, mode: str = 'human', **kwargs) -> Optional[np.ndarray]:
        """Render the robot at current sate.
        """
        if not self.simulator.is_viewer_available:
            kwargs["camera_xyzrpy"] = [(0.0, 7.0, 0.0), (np.pi/2, 0.0, np.pi)]
        return super().render(mode, **kwargs)


class AcrobotJiminyGoalEnv(AcrobotJiminyEnv, BaseJiminyGoalEnv):
    """ TODO: Write documentation.
    """
    def _get_goal_space(self) -> gym.Space:
        """ TODO: Write documentation.
        """
        return gym.spaces.Box(
            low=-self._tipPosZMax,
            high=self._tipPosZMax,
            dtype=np.float64)

    def _sample_goal(self) -> np.ndarray:
        """Sample goal.

        The goal is sampled using a uniform distribution in range
        [HEIGHT_REL_MIN_GOAL_THRESHOLD, HEIGHT_REL_MAX_GOAL_THRESHOLD].
        """
        return self._tipPosZMax * sample(
            *HEIGHT_REL_GOAL_THRESHOLD_RANGE, shape=(1,), rg=self.rg)

    def _get_achieved_goal(self) -> np.ndarray:
        """Compute achieved goal based on current state of the robot.

        It corresponds to the position of the tip of the acrobot.
        """
        tip_transform = self.robot.pinocchio_data.oMf[self._tipIdx]
        tip_position_z = tip_transform.translation[2]
        return np.array([tip_position_z])

    def is_done(self,  # type: ignore[override]
                achieved_goal: Optional[np.ndarray] = None,
                desired_goal: Optional[np.ndarray] = None) -> bool:
        """Determine whether a desired goal has been achieved.

        The episode is successful if the achieved goal strictly exceeds the
        desired goal.
        """
        # pylint: disable=arguments-differ

        if achieved_goal is None:
            achieved_goal = self._get_achieved_goal()
        if desired_goal is None:
            desired_goal = self._desired_goal
        return bool(achieved_goal > desired_goal)

    def compute_reward(self,  # type: ignore[override]
                       achieved_goal: Optional[np.ndarray] = None,
                       desired_goal: Optional[np.ndarray] = None,
                       *, info: Dict[str, Any]) -> float:
        """Compute reward at current episode state.

        Get a small negative reward till success.
        """
        # pylint: disable=arguments-differ

        if self.is_done(achieved_goal, desired_goal):
            reward = 0.0
        else:
            reward = -1.0
        return reward
