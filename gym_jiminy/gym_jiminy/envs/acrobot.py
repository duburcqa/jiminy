## @file

import os
import numpy as np
from pkg_resources import resource_filename
from typing import Optional, Tuple, Dict, Any

from gym import spaces

from jiminy_py import core as jiminy
from jiminy_py.simulator import Simulator

from ..common.env_bases import SpaceDictRecursive, BaseJiminyGoalEnv


DT = 2.0e-3          # Stepper update period
MAX_VEL = 4 * np.pi  # Max velocity of the joints
MAX_TORQUE = 10.0    # Max torque of the motor
ACTION_NOISE = 0.0   # Standard deviation of the noise added to the action


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
                [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
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
        @brief      Constructor

        @param[in]  continuous   Whether or not the action space is continuous.
                                 If not continuous, the action space has only 3
                                 states, i.e. low, zero, and high.
                                 Optional: True by default.
        """
        # Backup some input arguments
        self.continuous = continuous

        # Initialize Jiminy simulator

        ## Get URDF path
        data_dir = resource_filename('gym_jiminy.envs', 'data/toys_models')
        urdf_path = os.path.join(
            data_dir, "double_pendulum/double_pendulum.urdf")

        ## Instantiate robot
        robot = jiminy.Robot()
        robot.initialize(urdf_path,
            has_freeflyer=False, mesh_package_dirs=[data_dir])

        ## Add motors and sensors
        motor_joint_name = "SecondPendulumJoint"
        encoder_joint_names = ("PendulumJoint", "SecondPendulumJoint")
        motor = jiminy.SimpleMotor(motor_joint_name)
        robot.attach_motor(motor)
        motor.initialize(motor_joint_name)
        for joint_name in encoder_joint_names:
            encoder = jiminy.EncoderSensor(joint_name)
            robot.attach_sensor(encoder)
            encoder.initialize(joint_name)

        ## Instantiate simulator
        simulator = Simulator(robot)

        # Define some problem-specific variables

        ## Map between discrete actions and actual motor torque if necessary
        if not self.continuous:
            self.AVAIL_TORQUE = [-MAX_TORQUE, MAX_TORQUE]

        ## Angle at which to fail the episode
        self.theta_threshold_radians = 25 * np.pi / 180

        ## Position at which to fail the episode
        self.x_threshold = 0.75

        ## Internal parameters used for sampling goals and computing
        #  termination condition.
        self._tipIdx = robot.pinocchio_model.getFrameId("SecondPendulumMass")
        self._tipPosZMax = robot.pinocchio_data.oMf[
            self._tipIdx].translation[2]

        ## Bounds of hypercube for initial state uniform sampling
        self.position_random_high = np.array([ 0.2 - np.pi,  0.2])
        self.position_random_low  = np.array([-0.2 - np.pi, -0.2])
        self.velocity_random_high = np.full((2,), 1.0)
        self.velocity_random_low  = -self.velocity_random_high

        # Configure the learning environment
        super().__init__(simulator, DT, debug=False)

    def _setup_environment(self) -> None:
        """
        @brief    TODO
        """
        super()._setup_environment()

        robot_options = self.robot.get_options()

        # Set the position and velocity bounds of the robot
        robot_options["model"]["joints"]["velocityLimitFromUrdf"] = False
        robot_options["model"]["joints"]["velocityLimit"] = np.full(2, MAX_VEL)

        # Set the effort limit of the motor
        motor_name = self.robot.motors_names[0]
        robot_options["motors"][motor_name]["effortLimitFromUrdf"] = False
        robot_options["motors"][motor_name]["effortLimit"] = MAX_TORQUE

        self.robot.set_options(robot_options)

    def _refresh_observation_space(self) -> None:
        """
        @brief Configure the observation of the environment.

        @details Implement the official Gym acrobot-v1 action space. See
                 documentation https://gym.openai.com/envs/Acrobot-v1/.
        """
        # Compute observation and goal bounds
        obs_high = np.array([1.0, 1.0, 1.0, 1.0, 1.5 * MAX_VEL, 1.5 * MAX_VEL])
        goal_high = np.array([self._tipPosZMax])

        # Set the observation space, gathering the actual observation and the
        # goal subspaces.
        self.observation_space = spaces.Dict(
            desired_goal=spaces.Box(
                low=-goal_high, high=goal_high, dtype=np.float64),
            achieved_goal=spaces.Box(
                low=-goal_high, high=goal_high, dtype=np.float64),
            observation=spaces.Box(
                low=-obs_high, high=obs_high, dtype=np.float64))

        # Reset observation
        self._observation = {
            'observation': None, 'achieved_goal': None, 'desired_goal': None}

    def _refresh_action_space(self) -> None:
        """
        @brief    TODO

        @details Replace the action space by its discrete representation
                 depending on 'continuous'.
        """
        if not self.continuous:
            self.action_space = spaces.Discrete(2)
        else:
            super()._refresh_action_space()

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        @brief    TODO
        """
        qpos = self.rg.uniform(low=self.position_random_low,
                               high=self.position_random_high)
        qvel = self.rg.uniform(low=self.velocity_random_low,
                               high=self.velocity_random_high)
        return qpos, qvel

    def _sample_goal(self) -> np.ndarray:
        """
        @brief    TODO
        """
        return self.rg.uniform(low=-0.20*self._tipPosZMax,
                               high=0.98*self._tipPosZMax,
                               size=(1,))

    def _get_achieved_goal(self) -> np.ndarray:
        """
        @brief    TODO
        """
        tip_transform = self.robot.pinocchio_data.oMf[self._tipIdx]
        tip_position_z = tip_transform.translation[2]
        return np.array([tip_position_z])

    def _fetch_obs(self) -> SpaceDictRecursive:
        # @copydoc BaseJiminyEnv::_fetch_obs
        (theta1, theta2), (theta1_dot, theta2_dot) = self.simulator.state
        obs = {}
        obs['observation'] = np.array([np.cos(theta1 + np.pi),
                                       np.sin(theta1 + np.pi),
                                       np.cos(theta2 + np.pi),
                                       np.sin(theta2 + np.pi),
                                       theta1_dot,
                                       theta2_dot])
        obs['achieved_goal'] = self._get_achieved_goal()
        obs['desired_goal'] = self._desired_goal.copy()
        return obs

    def _is_done(self,
                 achieved_goal: Optional[np.ndarray] = None,
                 desired_goal: Optional[np.ndarray] = None) -> bool:
        """
        @brief    TODO
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
        @brief    TODO
        """
        # Check if the desired goal has been achieved
        done = self._is_done(achieved_goal, desired_goal)

        # Get a negative reward till success
        reward = 0.0
        if not done:
            reward += -1.0 #-self.dt # For the cumulative reward to be invariant wrt the simulation timestep
        return reward

    def step(self, action: Optional[np.ndarray] = None
            ) -> Tuple[SpaceDictRecursive, float, bool, Dict[str, Any]]:
        """
        @brief    TODO
        """
        if action is not None:
            # Make sure that the action is not oyt-of-bounds
            assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))

            # Compute the actual torque to apply
            if not self.continuous:
                action = self.AVAIL_TORQUE[action]
            if ACTION_NOISE > 0.0:
                action += self.rg.uniform(-ACTION_NOISE, ACTION_NOISE)

        # Perform the step
        return super().step(action)

    def render(self, mode: str = 'human', **kwargs) -> Optional[np.ndarray]:
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
    def __init__(self, continuous: bool = True, enableGoalEnv: bool = False):
        """
        @brief    TODO

        @params enableGoalEnv  Whether or not goal is enable.
        """
        self.enableGoalEnv = enableGoalEnv
        super().__init__(continuous)

    def _refresh_observation_space(self) -> None:
        """
        @brief    TODO
        """
        super()._refresh_observation_space()
        if not self.enableGoalEnv:
            self.observation_space = self.observation_space['observation']
            self._observation = self._observation['observation']

    def _sample_goal(self) -> np.ndarray:
        """
        @brief    TODO
        """
        if self.enableGoalEnv:
            return super()._sample_goal()
        else:
            return np.array([0.95 * self._tipPosZMax])

    def _fetch_obs(self) -> SpaceDictRecursive:
        # @copydoc BaseJiminyEnv::_fetch_obs
        obs = super()._fetch_obs()
        if self.enableGoalEnv:
            return obs
        else:
            return obs['observation']
