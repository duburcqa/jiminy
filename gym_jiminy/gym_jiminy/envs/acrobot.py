## @file

import os
import numpy as np
from math import sin, cos, pi
from pkg_resources import resource_filename

from gym import core, spaces, logger
from gym.utils import seeding

from jiminy_py import core as jiminy
from jiminy_py.engine_asynchronous import EngineAsynchronous

from ..common.robots import RobotJiminyEnv, RobotJiminyGoalEnv

DT = 2.0e-3         ## Stepper update period
MAX_VEL = 4 * pi    ## Max velocity of the joints
MAX_TORQUE = 10.0   ## Max torque of the motor
ACTION_NOISE = 0.0  ## Standard deviation of the noise added to the action

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

    ## @var metadata
    # @copydoc RobotJiminyGoalEnv::metadata

    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, continuous=True):
        """
        @brief      Constructor

        @param[in]  continuous      Whether the action space is continuous or not. If
                                    not continuous, the action space has only 3 states,
                                    i.e. low, zero, and high. Optional: True by default

        @return     Instance of the environment.
        """

        ## @var steps_beyond_done
        # @copydoc RobotJiminyGoalEnv::steps_beyond_done
        ## @var state
        # @copydoc RobotJiminyGoalEnv::state
        ## @var action_space
        #  @copydoc RobotJiminyGoalEnv::action_space
        ## @var observation_space
        # @copydoc RobotJiminyGoalEnv::observation_space
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

        self._robot = jiminy.Robot() # Robot has to be an attribute of the class to avoid being garbage collected
        self._robot.initialize(urdf_path)

        motor_joint_name = "SecondPendulumJoint"
        encoder_joint_names = ("PendulumJoint", "SecondPendulumJoint")
        motor = jiminy.SimpleMotor(motor_joint_name)
        self._robot.attach_motor(motor)
        motor.initialize(motor_joint_name)
        for joint_name in encoder_joint_names:
            encoder = jiminy.EncoderSensor(joint_name)
            self._robot.attach_sensor(encoder)
            encoder.initialize(joint_name)

        engine_py = EngineAsynchronous(self._robot)

        # ############################### Configure Jiminy #####################################

        robot_options = self._robot.get_options()

        # Set the position and velocity bounds of the robot
        robot_options["model"]["joints"]["velocityLimitFromUrdf"] = False
        robot_options["model"]["joints"]["velocityLimit"] = MAX_VEL * np.ones(2)

        # Set the effort limit of the motor
        robot_options["motors"][motor_joint_name]["effortLimitFromUrdf"] = False
        robot_options["motors"][motor_joint_name]["effortLimit"] = MAX_TORQUE

        self._robot.set_options(robot_options)

        # ##################### Define some problem-specific variables #########################

        if not self.continuous:
            ## Map between discrete actions and actual motor torque
            self.AVAIL_TORQUE = [-MAX_TORQUE, 0.0, MAX_TORQUE]

        ## Angle at which to fail the episode
        self.theta_threshold_radians = 25 * pi / 180

        ## Position at which to fail the episode
        self.x_threshold = 0.75

        # Internal parameters use to generate sample goals and compute the terminal condition
        self._tipIdx = engine_py._engine.robot.pinocchio_model.getFrameId("SecondPendulumMass")
        self._tipPosZMax = engine_py._engine.robot.pinocchio_data.oMf[self._tipIdx].translation[2]

        # ####################### Configure the learning environment ###########################

        super().__init__("acrobot", engine_py, DT)

        # #################### Overwrite some problem-generic variables ########################

        # Replace the observation space, which is NOT the sensor space in this case,
        # for consistency with the official gym acrobot-v1 (https://gym.openai.com/envs/Acrobot-v1/)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0, MAX_VEL, MAX_VEL])

        # Set bounds to the goal spaces, since they are known in this case (infinite by default)
        goal_high = np.array([self._tipPosZMax])

        self.observation_space = spaces.Dict(
            desired_goal=spaces.Box(low=-goal_high, high=goal_high, dtype=np.float64),
            achieved_goal=spaces.Box(low=-goal_high, high=goal_high, dtype=np.float64),
            observation=spaces.Box(low=-obs_high, high=obs_high, dtype=np.float64))

        # Bounds of the hypercube associated with the initial state of the robot
        self.state_random_high = np.array([ 0.2 - pi,  0.2,  1.0,  1.0])
        self.state_random_low  = np.array([-0.2 - pi, -0.2, -1.0, -1.0])

        if not self.continuous:
            self.action_space = spaces.Discrete(3)


    def _sample_state(self):
        """
        @brief      Returns a random valid initial state.
        """
        return self.np_random.uniform(low=self.state_random_low,
                                      high=self.state_random_high)


    def _sample_goal(self):
        """
        @brief      Samples a new goal and returns it.

        @details    The goal is randomly sampled using a uniform
                    distribution between `0.2*self._tipPosZMax` and
                    `0.98*self._tipPosZMax`.

        @remark     `self._tipPosZMax` can be overwritten to tweak the
                    difficulty of the problem. By default, it is the
                    maximum high of the highest joint that can be reached
                    by the pendulum.

        @return     Sample goal.
        """
        return self.np_random.uniform(low=-0.2*self._tipPosZMax,
                                      high=0.98*self._tipPosZMax,
                                      size=(1,))


    def step(self, action):
        """
        @brief      Run a simulation step for a given.

        @param[in]  action   The action to perform (in the action space rather than
                             the original torque space).

        @return     The next observation, the reward, the status of the simulation
                    (done or not), and a dictionary of extra information
        """

        # Compute the torque to apply
        if self.continuous:
            torque = action
        else:
            torque = self.AVAIL_TORQUE[action]

        if ACTION_NOISE > 0:
            torque += self.np_random.uniform(-ACTION_NOISE, ACTION_NOISE)

        # Bypass 'self.engine_py.step' method and use direct assignment to max out the performances
        self.engine_py._action[0] = torque
        self.engine_py.step(dt_desired=self.dt)
        self.state = self.engine_py.state

        # Get information
        info, obs = self._get_info()
        done = info['is_success']

        # Make sure the simulation is not already over
        if done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1

        # Compute the reward
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        return obs, reward, done, info


    def _get_info(self):
        """
        @brief      Get the observation associated with the current state of the robot,
                    along with some additional information.

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.

        @return     The next observation and a dictionary of extra information
        """

        # Get observation about the current state
        obs = self._get_obs()

        # Check the terminal condition
        done = self._is_success(obs['achieved_goal'], self.goal)

        # Generate info dict
        info = {'is_success': done}

        return info, obs


    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        @brief      Compute the reward at the current state.

        @param[in]  achieved_goal   Currently achieved goal
        @param[in]  desired_goal    Desired goal
        @param[in]  info            Dictionary of extra information

        @return     The computed reward.
        """

        # Must NOT use info, since it is not available while using HER (Experience Replay)

        reward = 0.0

        # Recompute the info if not available
        done = self._is_success(achieved_goal, desired_goal)

        # Get a negative reward till success
        if not done:
            reward += -1.0

        return reward


    def _get_achieved_goal(self):
        """
        @brief      Compute the achieved goal based on the current state of the robot.

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.

        @return     The currently achieved goal
        """
        return self.engine_py._engine.robot.pinocchio_data.oMf[self._tipIdx].translation[[2]]


    def _is_success(self, achieved_goal, desired_goal):
        """
        @brief      Determine whether the desired goal has been achieved.

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.

        @return     Boolean flag
        """
        return bool(achieved_goal > desired_goal)


    def _get_obs(self):
        """
        @brief      Get the current observation based on the current state of the robot.

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.

        @return     Dictionary with the current observation, achieved goal,
                    and desired goal.
        """
        theta1, theta2, theta1_dot, theta2_dot  = self.state
        observation = np.array([cos(theta1 + pi),
                                sin(theta1 + pi),
                                cos(theta2 + pi),
                                sin(theta2 + pi),
                                theta1_dot,
                                theta2_dot])

        achieved_goal = self._get_achieved_goal()
        return {
            'observation': observation,
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }


class JiminyAcrobotEnv(JiminyAcrobotGoalEnv):
    """
    @brief      Implementation of a Gym goal-environment for the Acrobot which is using
                Jiminy Engine to perform physics computations and Gepetto-viewer for
                rendering.

    @details    It only changes the observation mechanism wrt the base class
                `JiminyAcrobotGoalEnv`. See its documentation for more information.
    """

    ## @var observation_space
    # @copydoc RobotJiminyGoalEnv::observation_space

    def __init__(self, continuous=True, enableGoalEnv=False):
        """
        @brief      Constructor

        @param[in]  continuous      Whether the action space is continuous or not. If
                                    not continuous, the action space has only 3 states,
                                    i.e. low, zero, and high. Optional: True by default
        @param[in]  enableGoalEnv   Whether the goal is actually enable or not.
                                    Optional: False by default

        @return     Instance of the environment.
        """

        ## Flag to determine if the goal is enable
        self.enableGoalEnv = enableGoalEnv

        super().__init__(continuous)

        if not self.enableGoalEnv:
            self.observation_space = self.observation_space['observation']


    def _sample_goal(self):
        """
        @brief      Samples a new goal and returns it.

        @details    Actually, it samples a new goal only if `enableGoalEnv=True`.
                    Otherwise, the goal is always 95% of the maximum height that
                    can possibly be reached by the highest point of the Acrobot.

        @remark     See documentation of `JiminyAcrobotEnv._sample_goal` for details.

        @return     Sample goal.
        """
        if self.enableGoalEnv:
            return super()._sample_goal()
        else:
            return np.array([0.95 * self._tipPosZMax])


    def reset(self):
        """
        @brief      Reset the simulation.

        @remark     See documentation of `RobotJiminyGoalEnv` for details.

        @return     Initial state of the simulation
        """
        obs = super().reset()
        if self.enableGoalEnv:
            return obs
        else:
            return obs['observation']


    def step(self, a):
        """
        @brief      Run a simulation step for a given.

        @remark     See documentation of `JiminyAcrobotGoalEnv` for details.

        @param[in]  a       The action to perform (in the action space rather than
                            the original torque space).

        @return     The next observation, the reward, the status of the simulation
                    (done or not), and a dictionary of extra information
        """
        obs, reward, done, info = super().step(a)
        if self.enableGoalEnv:
            return obs, reward, done, info
        else:
            return obs['observation'], reward, done, info
