## @file

import os
import numpy as np
from pkg_resources import resource_filename

from gym import spaces

from jiminy_py import core as jiminy
from jiminy_py.engine_asynchronous import EngineAsynchronous

from ..common.robots import RobotJiminyEnv, RobotJiminyGoalEnv


DT = 2.0e-3          ## Stepper update period
MAX_VEL = 4 * np.pi  ## Max velocity of the joints
MAX_TORQUE = 10.0    ## Max torque of the motor
ACTION_NOISE = 0.0   ## Standard deviation of the noise added to the action


class JiminyAcrobotGoalEnv(RobotJiminyGoalEnv):
    """
    @brief      Implementation of a Gym environment for the Acrobot which is using
                Jiminy Engine to perform physics computations and Gepetto-viewer for
                rendering. It is a specialization of RobotJiminyGoalEnv. The acrobot
                is a 2-link pendulum with only the second joint actuated. Initially,
                both links point downwards. The goal is to swing the end-effector at
                a height at least the length of one link above the base. Both links
                can swing freely and can pass by each other, i.e. they don't collide
                when they have the same angle.

    @details    **STATE:**
                The state consists of the sin() and cos() of the two rotational joint
                angles and the joint angular velocities :
                [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
                For the first link, an angle of 0 corresponds to the link pointing
                downwards. The angle of the second link is relative to the angle of
                the first link. An angle of 0 corresponds to having the same angle
                between the two links. A state of [1, 0, 1, 0, ..., ...] means that
                both links point downwards.

                **ACTIONS:**
                The action is either applying +1, 0 or -1 torque on the joint between
                the two pendulum links.

    @see        R. Sutton: Generalization in Reinforcement Learning:
                    Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    @see        R. Sutton and A. G. Barto:
                    Reinforcement learning: An introduction.
                    Cambridge: MIT press, 1998.
    """
    def __init__(self, continuous=False):
        """
        @brief      Constructor

        @param[in]  continuous      Whether the action space is continuous or not. If
                                    not continuous, the action space has only 3 states,
                                    i.e. low, zero, and high. Optional: True by default

        @return     Instance of the environment.
        """

        #  @copydoc RobotJiminyEnv::__init__
        ## @var state_random_high
        #  @copydoc RobotJiminyGoalEnv::state_random_high
        ## @var state_random_low
        #  @copydoc RobotJiminyGoalEnv::state_random_low

        # ########################## Backup the input arguments ################################

        ## Flag to determine if the action space is continuous
        self.continuous = continuous

        # ############################### Initialize Jiminy ####################################

        os.environ["JIMINY_MESH_PATH"] = resource_filename('gym_jiminy.envs', 'data')
        urdf_path = os.path.join(os.environ["JIMINY_MESH_PATH"], "double_pendulum/double_pendulum.urdf")

        robot = jiminy.Robot()
        robot.initialize(urdf_path)

        motor_joint_name = "SecondPendulumJoint"
        encoder_joint_names = ("PendulumJoint", "SecondPendulumJoint")
        motor = jiminy.SimpleMotor(motor_joint_name)
        robot.attach_motor(motor)
        motor.initialize(motor_joint_name)
        for joint_name in encoder_joint_names:
            encoder = jiminy.EncoderSensor(joint_name)
            robot.attach_sensor(encoder)
            encoder.initialize(joint_name)

        engine_py = EngineAsynchronous(robot)

        # ##################### Define some problem-specific variables #########################

        if not self.continuous:
            ## Map between discrete actions and actual motor torque
            self.AVAIL_TORQUE = [-MAX_TORQUE, MAX_TORQUE]

        ## Angle at which to fail the episode
        self.theta_threshold_radians = 25 * np.pi / 180

        ## Position at which to fail the episode
        self.x_threshold = 0.75

        # Internal parameters use to generate sample goals and compute the terminal condition
        self._tipIdx = robot.pinocchio_model.getFrameId("SecondPendulumMass")
        self._tipPosZMax = robot.pinocchio_data.oMf[self._tipIdx].translation[2]

        # Bounds of the hypercube associated with the initial state of the robot
        self.state_random_high = np.array([ 0.2 - np.pi,  0.2,  1.0,  1.0])
        self.state_random_low  = np.array([-0.2 - np.pi, -0.2, -1.0, -1.0])

        # ####################### Configure the learning environment ###########################

        super().__init__("acrobot", engine_py, DT)

    def _setup_environment(self):
        super()._setup_environment()

        # Override some options of the robot and engine
        robot_options = self.robot.get_options()

        ### Set the position and velocity bounds of the robot
        robot_options["model"]["joints"]["velocityLimitFromUrdf"] = False
        robot_options["model"]["joints"]["velocityLimit"] = MAX_VEL * np.ones(2)

        ### Set the effort limit of the motor
        motor_name = self.robot.motors_names[0]
        robot_options["motors"][motor_name]["effortLimitFromUrdf"] = False
        robot_options["motors"][motor_name]["effortLimit"] = MAX_TORQUE

        self.robot.set_options(robot_options)

    def _refresh_learning_spaces(self):
        # Replace the observation space, which is NOT the sensor space in this case,
        # for consistency with the official gym acrobot-v1 (https://gym.openai.com/envs/Acrobot-v1/)
        super()._refresh_learning_spaces()

        # Replace the action space if necessary
        if not self.continuous:
            self.action_space = spaces.Discrete(2)

        # Set bounds to the goal spaces, since they are known in this case (infinite by default)
        goal_high = np.array([self._tipPosZMax])

        obs_high = np.array([1.0, 1.0, 1.0, 1.0, 1.5 * MAX_VEL, 1.5 * MAX_VEL])

        self.observation_space = spaces.Dict(
            desired_goal=spaces.Box(low=-goal_high, high=goal_high, dtype=np.float64),
            achieved_goal=spaces.Box(low=-goal_high, high=goal_high, dtype=np.float64),
            observation=spaces.Box(low=-obs_high, high=obs_high, dtype=np.float64))

        ## Current observation of the robot
        self.observation = {'observation': None,
                            'achieved_goal': None,
                            'desired_goal': None}

    def _sample_state(self):
        # @copydoc RobotJiminyEnv::_sample_state
        return self.rg.uniform(low=self.state_random_low,
                               high=self.state_random_high)

    def _sample_goal(self):
        # @copydoc RobotJiminyGoalEnv::_sample_goal
        return self.rg.uniform(low=-0.20*self._tipPosZMax,
                               high=0.98*self._tipPosZMax,
                               size=(1,))

    def _get_achieved_goal(self):
        # @copydoc RobotJiminyGoalEnv::_get_achieved_goal
        return self.robot.pinocchio_data.oMf[self._tipIdx].translation[[2]].copy()

    def _update_obs(self, obs):
        # @copydoc RobotJiminyEnv::_update_observation
        theta1, theta2, theta1_dot, theta2_dot  = self.engine_py.state
        obs['observation'] = np.array([np.cos(theta1 + np.pi),
                                       np.sin(theta1 + np.pi),
                                       np.cos(theta2 + np.pi),
                                       np.sin(theta2 + np.pi),
                                       theta1_dot,
                                       theta2_dot])
        obs['achieved_goal'] = self._get_achieved_goal()
        obs['desired_goal'] = self.goal.copy()

    def _is_done(self, achieved_goal=None, desired_goal=None):
        # @copydoc RobotJiminyGoalEnv::_is_done
        if achieved_goal is None:
            achieved_goal = self.observation['achieved_goal']
        if desired_goal is None:
            desired_goal = self.observation['desired_goal']
        return bool(achieved_goal > desired_goal)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # @copydoc RobotJiminyGoalEnv::compute_reward

        # Check if the desired goal has been achieved
        done = self._is_done(achieved_goal, desired_goal)

        # Get a negative reward till success
        reward = 0.0
        if not done:
            reward += -1.0 #-self.dt # For the cumulative reward to be invariant wrt the simulation timestep
        return reward

    def step(self, action):
        # @copydoc RobotJiminyEnv::step
        if action is not None:
            # Make sure that the action is within bounds
            assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

            # Compute the actual torque to apply
            if not self.continuous:
                action = self.AVAIL_TORQUE[action]
            if ACTION_NOISE > 0.0:
                action += self.rg.uniform(-ACTION_NOISE, ACTION_NOISE)

        # Perform the step
        return super().step(action)


class JiminyAcrobotEnv(JiminyAcrobotGoalEnv):
    """
    @brief      Implementation of a Gym goal-environment for the Acrobot which is using
                Jiminy Engine to perform physics computations and Gepetto-viewer for
                rendering.

    @details    It only changes the observation mechanism wrt the base class
                `JiminyAcrobotGoalEnv`. See its documentation for more information.
    """
    def __init__(self, continuous=True, enableGoalEnv=False):
        # @copydoc RobotJiminyGoalEnv::__init__

        ## Flag to determine if the goal is enable
        self.enableGoalEnv = enableGoalEnv

        super().__init__(continuous)

    def _refresh_learning_spaces(self):
        super()._refresh_learning_spaces()
        if not self.enableGoalEnv:
            self.observation_space = self.observation_space['observation']

    def _sample_goal(self):
        # @copydoc RobotJiminyGoalEnv::_sample_goal
        if self.enableGoalEnv:
            return super()._sample_goal()
        else:
            return np.array([0.95 * self._tipPosZMax])

    def reset(self):
        # @copydoc RobotJiminyEnv::reset
        obs = super().reset()
        if self.enableGoalEnv:
            return obs
        else:
            return obs['observation']

    def step(self, a):
        # @copydoc RobotJiminyEnv::step
        obs, reward, done, info = super().step(a)
        if self.enableGoalEnv:
            return obs, reward, done, info
        else:
            return obs['observation'], reward, done, info
