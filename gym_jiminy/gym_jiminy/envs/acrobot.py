import os
import numpy as np
from pkg_resources import resource_filename
from typing import Optional, Tuple, Dict, Any

from gym import spaces

from jiminy_py import core as jiminy
from jiminy_py.simulator import Simulator

from ..common.env_bases import SpaceDictRecursive, BaseJiminyGoalEnv


# Stepper update period
STEP_DT = 0.2
# Range of uniform sampling distribution of joint angles
THETA_RANDOM_RANGE = 0.1
# Range of uniform sampling distribution of joint velocities
DTHETA_RANDOM_RANGE = 0.1
# Relative height of tip to consider an episode successful for normal env
HEIGHT_REL_DEFAULT_THRESHOLD = 0.5
# Mim relative height of tip to consider an episode successful for goal env
HEIGHT_REL_MIN_GOAL_THRESHOLD = -0.2
# Max relative height of tip to consider an episode successful for goal env
HEIGHT_REL_MAX_GOAL_THRESHOLD = 0.98
# Standard deviation of the noise added to the action
ACTION_NOISE = 0.0


class AcrobotJiminyGoalEnv(BaseJiminyGoalEnv):
    """
    @brief      Implementation of a Gym environment for the Acrobot which is
                using Jiminy Engine to perform physics computations and Meshcat
                for rendering.

    @remark     It is a specialization of BaseJiminyGoalEnv. The acrobot is a
                2-link pendulum with only the second joint actuated. Initially,
                both links point downwards. The goal is to swing the
                end-effector at a height at least the length of one link above
                the base. Both links can swing freely and can pass by each
                other, i.e. they don't collide when they have the same angle.

    @details    **STATE:**
                The state consists of the sin() and cos() of the two rotational
                joint angles and the joint angular velocities :
                [cos(theta1) sin(theta1) cos(theta2) sin(theta2)
                 thetaDot1 thetaDot2].
                For the first link, an angle of 0 corresponds to the link
                pointing downwards. The angle of the second link is relative to
                the angle of the first link. An angle of 0 corresponds to
                having the same angle between the two links. A state of
                [1, 0, 1, 0, ..., ...] means that both links point downwards.

                **ACTIONS:**
                The action is either applying +1, 0 or -1 torque on the joint
                between the two pendulum links.

    @see        R. Sutton: Generalization in Reinforcement Learning:
                    Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    @see        R. Sutton and A. G. Barto:
                    Reinforcement learning: An introduction.
                    Cambridge: MIT press, 1998.
    """
    def __init__(self, continuous: bool = False):
        """
        @brief Constructor

        @param continuous  Whether or not the action space is continuous. If
                           not continuous, the action space has only 3 states,
                           i.e. low, zero, and high.
                           Optional: True by default.
        """
        # Backup some input arguments
        self.continuous = continuous

        # Get URDF path
        data_dir = resource_filename('gym_jiminy.envs', 'data/toys_models')
        urdf_path = os.path.join(
            data_dir, "acrobot/acrobot.urdf")

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
            self.AVAIL_TORQUE = [-motor.effort_limit, 0.0, motor.effort_limit]

        # Internal parameters used for sampling goals and computing termination
        # condition.
        self._tipIdx = robot.pinocchio_model.getFrameId("Tip")
        self._tipPosZMax = - robot.pinocchio_data.oMf[
            self._tipIdx].translation[2]

        # Configure the learning environment
        super().__init__(simulator, STEP_DT, debug=False)

    def _refresh_observation_space(self) -> None:
        """
        @brief Configure the observation of the environment.

        @details Only the state is observable, while by default, the current
                 time, state, and sensors data are available.
        """
        super()._refresh_observation_space()
        self.observation_space.spaces['observation'] = \
            self.observation_space['observation']['state']

    def _fetch_obs(self) -> None:
        """
        @brief Fetch the observation based on the current state of the robot.

        @details Only the state is observable, while by default, the current
                 time, state, and sensors data are available.

        @remark For goal env, both the desired and achieved goals are
                observable in in addition of the current robot state.
        """
        obs = super()._fetch_obs()
        obs['observation'] = obs['observation']['state']
        return obs

    def _refresh_action_space(self) -> None:
        """
        @brief Configure the action space of the environment.

        @details Replace the action space by its discrete representation
                 depending on 'continuous'.
        """
        if not self.continuous:
            self.action_space = spaces.Discrete(len(self.AVAIL_TORQUE))
        else:
            super()._refresh_action_space()

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        @brief Returns a valid configuration and velocity for the robot.

        @details The initial state is randomly sampled using a hypercube
                 uniform distribution, according to the official Gym
                 acrobot-v1 action space. See documentation
                 https://gym.openai.com/envs/Acrobot-v1/.
        """
        theta1, theta2 = self.rg.uniform(low=-THETA_RANDOM_RANGE,
                                         high=THETA_RANDOM_RANGE,
                                         size=(2,))
        qpos = np.array([np.cos(theta1), np.sin(theta1),
                         np.cos(theta2), np.sin(theta2)])
        qvel = self.rg.uniform(low=-DTHETA_RANDOM_RANGE,
                               high=DTHETA_RANDOM_RANGE,
                               size=(2,))
        return qpos, qvel

    def _sample_goal(self) -> np.ndarray:
        """
        @brief Sample goal.

        @detail The goal is sampled using a uniform distribution between
                [HEIGHT_REL_MIN_GOAL_THRESHOLD, HEIGHT_REL_MAX_GOAL_THRESHOLD].
        """
        return self.rg.uniform(
            low=HEIGHT_REL_MIN_GOAL_THRESHOLD,
            high=HEIGHT_REL_MAX_GOAL_THRESHOLD,
            size=(1,)) * self._tipPosZMax

    def _get_achieved_goal(self) -> np.ndarray:
        """
        @brief Compute achieved goal based on current state of the robot.

        @details It corresponds to the position of the tip of the acrobot.
        """
        tip_transform = self.robot.pinocchio_data.oMf[self._tipIdx]
        tip_position_z = tip_transform.translation[2]
        return np.array([tip_position_z])

    def _is_done(self,
                 achieved_goal: Optional[np.ndarray] = None,
                 desired_goal: Optional[np.ndarray] = None) -> bool:
        """
        @brief Determine whether a desired goal has been achieved.

        @details The episode is successful if the achieved goal strictly
                 exceeds the desired goal.
        """
        if achieved_goal is None:
            achieved_goal = self._get_achieved_goal()
        if desired_goal is None:
            desired_goal = self._desired_goal
        return bool(achieved_goal > desired_goal)

    def compute_reward(self,
                       achieved_goal: Optional[np.ndarray],
                       desired_goal: Optional[np.ndarray],
                       info: Dict[str, Any]) -> float:
        """
        @brief Compute reward at current episode state.

        @details Get a small negative reward till success.
        """
        if self._is_done(achieved_goal, desired_goal):
            reward = 0.0
        else:
            reward = -1.0
        return reward

    def step(self,
             action: Optional[np.ndarray] = None
             ) -> Tuple[SpaceDictRecursive, float, bool, Dict[str, Any]]:
        """
        @brief Run a simulation step for a given action.

        @details Convert a  discrete action into its actual value if necessary,
                 then add noise to the action is enable.
        """
        if action is not None:
            # Compute the actual torque to apply
            if not self.continuous:
                action = self.AVAIL_TORQUE[action]
            if ACTION_NOISE > 0.0:
                action += self.rg.uniform(-ACTION_NOISE, ACTION_NOISE)

        # Perform the step
        return super().step(action)

    def render(self, mode: str = 'human', **kwargs) -> Optional[np.ndarray]:
        """
        @brief Render the current state of the robot.
        """
        if not self.simulator._is_viewer_available:
            kwargs["camera_xyzrpy"] = [(0.0, 7.0, 0.0), (np.pi/2, 0.0, np.pi)]
        return super().render(mode, **kwargs)


class AcrobotJiminyEnv(AcrobotJiminyGoalEnv):
    """
    @brief      Implementation of a Gym goal-environment for the Acrobot which
                is using Jiminy Engine to perform physics computations and
                Meshcat for rendering.

    @details    It only changes the observation mechanism wrt the base class
                `AcrobotJiminyGoalEnv`. See its documentation for more
                information.
    """
    def __init__(self, continuous: bool = True, enable_goal_env: bool = False):
        """
        @brief Constructor

        @param continuous  Whether or not the action space is continuous. If
                           not continuous, the action space has only 3 states,
                           i.e. low, zero, and high.
                           Optional: True by default.
        @params enable_goal_env  Whether or not goal is enable.
        """
        self.enable_goal_env = enable_goal_env
        super().__init__(continuous)

    def _refresh_observation_space(self) -> None:
        """
        @brief Configure the observation of the environment.

        @details Only the state is observable, while by default, the current
                 time, state, and sensors data are available.
        """
        if self.enable_goal_env:
            super()._refresh_observation_space()
        else:
            self.observation_space = self._get_state_space()

    def _sample_goal(self) -> np.ndarray:
        """
        @brief Sample goal.

        @detail The goal is always the same, and proportional to
                HEIGHT_REL_DEFAULT_THRESHOLD.
        """
        if self.enable_goal_env:
            return super()._sample_goal()
        else:
            return HEIGHT_REL_DEFAULT_THRESHOLD * self._tipPosZMax

    def _fetch_obs(self) -> SpaceDictRecursive:
        """
        @brief Fetch the observation based on the current state of the robot.

        @details Only the state is observed.
        """
        obs = super()._fetch_obs()
        if self.enable_goal_env:
            return obs
        else:
            return obs['observation']
