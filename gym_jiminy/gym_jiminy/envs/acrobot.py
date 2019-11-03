"""
Classic acrobot system simulated using Jiminy Engine
"""

import os
from math import sin, cos, pi
import numpy as np

from gym import core, spaces, logger
from gym.utils import seeding

import jiminy
from jiminy_py import engine_asynchronous
from gym_jiminy.common import RobotJiminyEnv, RobotJiminyGoalEnv


class JiminyAcrobotGoalEnv(RobotJiminyGoalEnv):
    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.

    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.

    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.

    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    """

    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, continuous=True):
        ############################ Backup the input arguments ################################

        self.continuous = continuous

        ################################# Initialize Jiminy ####################################

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        urdf_path = os.path.join(cur_dir, "../../../data/double_pendulum/double_pendulum.urdf")
        motors = ["SecondPendulumJoint"]
        self.model = jiminy.model() # Model has to be an attribute of the class to avoid it being garbage collected
        self.model.initialize(urdf_path, motors=motors)
        self.model.add_encoder_sensor(joint_name="PendulumJoint")
        self.model.add_encoder_sensor(joint_name="SecondPendulumJoint")
        engine_py = engine_asynchronous(self.model)

        ################################# Configure Jiminy #####################################

        model_options = self.model.get_model_options()
        sensors_options = self.model.get_sensors_options()
        engine_options = engine_py.get_engine_options()
        ctrl_options = engine_py.get_controller_options()

        model_options["telemetry"]["enableEncoderSensors"] = False
        engine_options["telemetry"]["enableConfiguration"] = False
        engine_options["telemetry"]["enableVelocity"] = False
        engine_options["telemetry"]["enableAcceleration"] = False
        engine_options["telemetry"]["enableCommand"] = False
        engine_options["telemetry"]["enableEnergy"] = False

        engine_options["stepper"]["solver"] = "runge_kutta_dopri5" # ["runge_kutta_dopri5", "explicit_euler"]

        self.model.set_model_options(model_options)
        self.model.set_sensors_options(sensors_options)
        engine_py.set_engine_options(engine_options)
        engine_py.set_controller_options(ctrl_options)

        ####################### Define some problem-specific variables #########################

        # Max velocity of the joints
        self.MAX_VEL_1 = 4 * pi
        self.MAX_VEL_2 = 4 * pi

        # Torque magnitude of the action
        if not self.continuous:
            self.AVAIL_TORQUE = [-1.0, 0.0, +1.0]

        # Force mag of the action
        self.torque_mag = np.array([10.0])

        # Noise standard deviation added to the action
        self.torque_noise_max = 0.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 25 * pi / 180
        self.x_threshold = 0.75

        # Internal parameters to generate sample goals and compute the terminal condition
        self._tipIdx = engine_py._engine.model.pinocchio_model.getFrameId("SecondPendulumMass")
        self._tipPosZMax = engine_py._engine.model.pinocchio_data.oMf[self._tipIdx].translation.A1[2]

        ######################### Configure the learning environment ###########################

        # The time step of the 'step' method
        dt = 2.0e-3

        super(JiminyAcrobotGoalEnv, self).__init__("acrobot", engine_py, dt)

        ###################### Overwrite some problem-generic variables ########################

        # Update the velocity bounds of the model
        model_options = self.model.get_model_options()
        model_options["joints"]["velocityLimit"] = [self.MAX_VEL_1, self.MAX_VEL_2]
        model_options["joints"]["velocityLimitFromUrdf"] = False
        self.model.set_model_options(model_options)

        # Update the goal spaces and the observation space (which is different from the state space in this case)
        goal_high = np.array([self._tipPosZMax])
        obs_high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(low=-goal_high, high=goal_high, dtype=np.float64),
            achieved_goal=spaces.Box(low=-goal_high, high=goal_high, dtype=np.float64),
            observation=spaces.Box(low=-obs_high, high=obs_high, dtype=np.float64)
        ))

        self.state_random_high = np.array([ 0.2 - pi,  0.2,  1.0,  1.0])
        self.state_random_low  = np.array([-0.2 - pi, -0.2, -1.0, -1.0])

        if self.continuous:
            self.action_space = spaces.Box(low=-self.torque_mag,
                                           high=self.torque_mag,
                                           dtype=np.float64)
        else:
            self.action_space = spaces.Discrete(3)

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        return self.np_random.uniform(low=-0.2*self._tipPosZMax, high=0.98*self._tipPosZMax, size=(1,))

    def step(self, a):
        if self.continuous:
            torque = a
        else:
            torque = self.AVAIL_TORQUE[a] * self.torque_mag

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

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
        # Get observation about the current state
        obs = self._get_obs()

        # Check the terminal condition
        done = self._is_success(obs['achieved_goal'], self.goal)

        # Generate info dict
        info = {'is_success': done}

        return info, obs

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Must NOT use info, since it is not available while using HER (Experience Replay)

        reward = 0.0

        # Recompute the info if not available
        done = self._is_success(achieved_goal, desired_goal)

        # Get a negative reward till success
        if not done:
            reward += -1.0

        return reward

    def _get_achieved_goal(self):
        return self.engine_py._engine.model.pinocchio_data.oMf[self._tipIdx].translation.A1[[2]]

    def _is_success(self, achieved_goal, desired_goal):
        return bool(achieved_goal > desired_goal)

    def _get_obs(self):
        theta1, theta2, theta1_dot, theta2_dot  = self.state
        theta1_dot = min(max(theta1_dot, -self.MAX_VEL_1), self.MAX_VEL_1)
        theta2_dot = min(max(theta2_dot, -self.MAX_VEL_2), self.MAX_VEL_2)
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
    def __init__(self, continuous=True, enableGoalEnv=False):
        self.enableGoalEnv = enableGoalEnv

        super(JiminyAcrobotEnv, self).__init__(continuous)

        if not self.enableGoalEnv:
            self.observation_space = self.observation_space['observation']

    def _sample_goal(self):
        if self.enableGoalEnv:
            return super(JiminyAcrobotEnv, self)._sample_goal()
        else:
            return np.array([0.95*self._tipPosZMax])

    def reset(self):
        obs = super(JiminyAcrobotEnv, self).reset()
        if self.enableGoalEnv:
            return obs
        else:
            return obs['observation']

    def step(self, a):
        obs, reward, done, info = super(JiminyAcrobotEnv, self).step(a)
        if self.enableGoalEnv:
            return obs, reward, done, info
        else:
            return obs['observation'], reward, done, info
