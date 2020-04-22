## @file

import os
import numpy as np
from math import sin, cos, pi
from pkg_resources import resource_filename

from gym import core, spaces, logger
from gym.utils import seeding

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

    ## @var metadata
    # @copydoc RobotJiminyEnv::metadata

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):
        """
        @brief      Constructor

        @return     Instance of the environment.
        """

        ## @var steps_beyond_done
        # @copydoc RobotJiminyEnv::steps_beyond_done
        ## @var state
        # @copydoc RobotJiminyEnv::state
        ## @var action_space
        #  @copydoc RobotJiminyEnv::action_space
        ## @var observation_space
        # @copydoc RobotJiminyEnv::observation_space
        ## @var state_random_high
        #  @copydoc RobotJiminyEnv::state_random_high
        ## @var state_random_low
        #  @copydoc RobotJiminyEnv::state_random_low

        # ############################### Initialize Jiminy ####################################

        os.environ["JIMINY_MESH_PATH"] = resource_filename('gym_jiminy.envs', 'data')
        urdf_path = os.path.join(os.environ["JIMINY_MESH_PATH"], "cartpole/cartpole.urdf")

        self._robot = jiminy.Robot()
        self._robot.initialize(urdf_path)

        motor_joint_name = "slider_to_cart"
        encoder_sensors_def = {"slider": "slider_to_cart", "pole": "cart_to_pole"}
        motor = jiminy.SimpleMotor(motor_joint_name)
        self._robot.attach_motor(motor)
        motor.initialize(motor_joint_name)
        for sensor_name, joint_name in encoder_sensors_def.items():
            encoder = jiminy.EncoderSensor(sensor_name)
            self._robot.attach_sensor(encoder)
            encoder.initialize(joint_name)

        engine_py = EngineAsynchronous(self._robot)

        # ############################### Configure Jiminy #####################################

        robot_options = self._robot.get_options()

        # Set the effort limit of the motor
        robot_options["motors"][motor_joint_name]["effortLimitFromUrdf"] = False
        robot_options["motors"][motor_joint_name]["effortLimit"] = MAX_FORCE

        self._robot.set_options(robot_options)

        # ##################### Define some problem-specific variables #########################

        ## Map between discrete actions and actual motor force
        self.AVAIL_FORCE = [-MAX_FORCE, MAX_FORCE]

        ## Maximum absolute angle of the pole before considering the episode failed
        self.theta_threshold_radians = 25 * pi / 180

        ## Maximum absolute position of the cart before considering the episode failed
        self.x_threshold = 0.75

        # ####################### Configure the learning environment ###########################

        super().__init__("cartpole", engine_py, DT)

        # #################### Overwrite some problem-generic variables ########################

        # Replace the observation space, which is the spa space instead of the sensor space.
        # Note that the Angle limit set to 2 * theta_threshold_radians, thus observations of
        # failure are still within bounds.
        high = np.array([self.x_threshold * 2,
                         self.theta_threshold_radians * 2,
                         *self._robot.velocity_limit])

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float64)

        # Bounds of the hypercube associated with the initial state of the robot
        self.state_random_high = np.array([0.5, 0.15, 0.1, 0.1])
        self.state_random_low = -self.state_random_high

        self.action_space = spaces.Discrete(2) # Force using a discrete action space


    def step(self, action):
        """
        @brief      Run a simulation step for a given.

        @param[in]  action  The action to perform (in the action space rather than
                            the original force space).

        @return     The next state, the reward, the status of the simulation (done or not),
                    and an empty dictionary for compatibility with Gym OpenAI.
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Compute the force to apply
        force = self.AVAIL_FORCE[action]

        # Bypass 'action' setter and use direct assignment to max out the performances
        self.engine_py._action[0] = force
        self.engine_py.step(dt_desired=self.dt)
        self.state = self.engine_py.state

        # Check the terminal condition and compute reward
        done = self._is_success()
        reward = 0.0
        if not done:
            reward += 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward += 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1

        return self.state, reward, done, {}


    def _sample_state(self):
        """
        @brief      Returns a random valid initial state.
        """
        return self.np_random.uniform(low=self.state_random_low,
                                      high=self.state_random_high)


    def _get_obs(self):
        """
        @brief      Get the current observation based on the current state of the robot.
                    Mostly defined for compatibility with Gym OpenAI.

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.

        @return     The current state of the robot
        """
        return self.state


    def _is_success(self):
        """
        @brief      Determine whether the desired goal has been achieved.

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.

        @return     Boolean flag
        """
        x, theta, _, _ = self.state
        return        x < -self.x_threshold \
               or     x >  self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta >  self.theta_threshold_radians
