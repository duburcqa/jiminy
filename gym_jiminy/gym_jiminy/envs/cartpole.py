## @file

import os
import numpy as np
from pkg_resources import resource_filename

from gym import spaces, logger

from jiminy_py import core as jiminy
from jiminy_py.engine_asynchronous import EngineAsynchronous

from ..common.robots import RobotJiminyEnv


DT = 2.0e-3      ## Stepper update period
MAX_FORCE = 40.0 ## Max force of the motor


class JiminyCartPoleEnv(RobotJiminyEnv):
    """
    @brief      Implementation of a Gym environment for the Cartpole which is using
                Jiminy Engine to perform physics computations and Gepetto-viewer for
                rendering. It is a specialization of RobotJiminyGoalEnv. The Cartpole
                is a pole attached by an un-actuated joint to a cart. The goal is to
                prevent the pendulum from falling over by increasing and reducing the
                cart's velocity.

    @details    **OBSERVATION:**
                Type: Box(4)
                Num	Observation                Min         Max
                0	Cart Position             -1.5         1.5
                1	Cart Velocity             -Inf         Inf
                2	Pole Angle                -50 deg      50 deg
                3	Pole Velocity At Tip      -Inf         Inf

                **ACTIONS:**
                Type: Discrete(2)
                Num	Action
                0	Push cart to the left
                1	Push cart to the right

                Note that the amount the velocity that is reduced or increased is not
                fixed, it depends on the angle the pole is pointing. This is because the
                center of gravity of the pole increases the amount of energy needed to
                move the cart underneath it.

                **REWARD:**
                Reward is 1 for every step taken, including the termination step.
                move the cart underneath it.

                **STARTING STATE:**
                All observations are assigned a uniform random value in [-0.05..0.05]

                **EPISODE TERMINATION:**
                If any of these conditions is satisfied:
                    - Pole Angle is more than 25 degrees Cart
                    - Position is more than 75 cm
                    - Episode length is greater than 200

                **SOLVED REQUIREMENTS:**
                Considered solved when the average reward is greater than or equal to
                195.0 over 100 consecutive trials.
    """
    def __init__(self, continuous=False):
        """
        @brief      Constructor

        @return     Instance of the environment.
        """

        #  @copydoc RobotJiminyEnv::__init__
        # ## @var state_random_high
        #  @copydoc RobotJiminyEnv::state_random_high
        ## @var state_random_low
        #  @copydoc RobotJiminyEnv::state_random_low

        # ########################## Backup the input arguments ################################

        ## Flag to determine if the action space is continuous
        self.continuous = continuous

        # ############################### Initialize Jiminy ####################################

        os.environ["JIMINY_MESH_PATH"] = resource_filename('gym_jiminy.envs', 'data')
        urdf_path = os.path.join(os.environ["JIMINY_MESH_PATH"], "cartpole/cartpole.urdf")

        robot = jiminy.Robot()
        robot.initialize(urdf_path)

        motor_joint_name = "slider_to_cart"
        encoder_sensors_def = {"slider": "slider_to_cart", "pole": "cart_to_pole"}
        motor = jiminy.SimpleMotor(motor_joint_name)
        robot.attach_motor(motor)
        motor.initialize(motor_joint_name)
        for sensor_name, joint_name in encoder_sensors_def.items():
            encoder = jiminy.EncoderSensor(sensor_name)
            robot.attach_sensor(encoder)
            encoder.initialize(joint_name)

        engine_py = EngineAsynchronous(robot)

        # ##################### Define some problem-specific variables #########################

        if not self.continuous:
            ## Map between discrete actions and actual motor force
            self.AVAIL_FORCE = [-MAX_FORCE, MAX_FORCE]

        ## Maximum absolute angle of the pole before considering the episode failed
        self.theta_threshold_radians = 25 * np.pi / 180

        ## Maximum absolute position of the cart before considering the episode failed
        self.x_threshold = 0.75

        # Bounds of the hypercube associated with the initial state of the robot
        self.state_random_high = np.array([0.5, 0.15, 0.1, 0.1])
        self.state_random_low = -self.state_random_high

        # ####################### Configure the learning environment ###########################

        super().__init__("cartpole", engine_py, DT)

    def _setup_environment(self):
        super()._setup_environment()

        # Override some options of the robot and engine
        robot_options = self.robot.get_options()

        ### Set the effort limit of the motor
        motor_name = self.robot.motors_names[0]
        robot_options["motors"][motor_name]["effortLimitFromUrdf"] = False
        robot_options["motors"][motor_name]["effortLimit"] = MAX_FORCE

        self.robot.set_options(robot_options)

    def _refresh_learning_spaces(self):
        # Replace the observation space, which is the state space instead of the sensor space.
        # Note that the Angle limit set to 2 * theta_threshold_radians, thus observations of
        # failure are still within bounds.
        super()._refresh_learning_spaces()

        # Force using a discrete action space
        if not self.continuous:
            self.action_space = spaces.Discrete(2)

        high = np.array([1.5 * self.x_threshold,
                         1.5 * self.theta_threshold_radians,
                         *self.robot.velocity_limit])

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float64)
        self.observation = np.zeros(self.observation_space.shape)

    def _sample_state(self):
        # @copydoc RobotJiminyEnv::_sample_state
        return self.rg.uniform(low=self.state_random_low,
                               high=self.state_random_high)

    def _update_obs(self, obs):
        # @copydoc RobotJiminyEnv::_update_observation
        obs[:] = self.engine_py.state

    @staticmethod
    def _key_to_action(key):
        if key == "Left":
            return 1
        elif key == "Right":
            return 0
        else:
            print(f"Key {key} is not bound to any action.")
            return None

    def _is_done(self):
        # @copydoc RobotJiminyEnv::_is_done
        x, theta, _, _ = self.observation
        return        x < -self.x_threshold \
               or     x >  self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta >  self.theta_threshold_radians

    def _compute_reward(self):
        # @copydoc RobotJiminyEnv::_compute_reward

        # Add a small positive reward as long as the terminal condition
        # has never been reached during the same episode.
        reward = 0.0
        if self._steps_beyond_done is None:
            done = self._is_done()
            if not done:
                reward += 1.0 #self.dt # For the cumulative reward to be invariant wrt the simulation timestep
        return reward

    def step(self, action):
        # @copydoc RobotJiminyEnv::step
        if action is not None:
            # Make sure that the action is within bounds
            assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

            # Compute the actual force to apply
            if not self.continuous:
                action = self.AVAIL_FORCE[action]

        # Perform the step
        obs, reward, done, info = super().step(action)

        # Update success flag, since in this case success actually
        # means never reaching terminal condition.
        info['is_success'] = not done

        return obs, reward, done, info
