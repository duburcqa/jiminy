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

        self._model = jiminy.Robot()
        self._model.initialize(urdf_path)

        motor_joint_names = ("slider_to_cart",)
        encoder_sensors_def = {"slider": "slider_to_cart", "pole": "cart_to_pole"}
        for joint_name in motor_joint_names:
            motor = jiminy.SimpleMotor(joint_name)
            self._model.attach_motor(motor)
            motor.initialize(joint_name)
        for sensor_name, joint_name in encoder_sensors_def.items():
            encoder = jiminy.EncoderSensor(sensor_name)
            self._model.attach_sensor(encoder)
            encoder.initialize(joint_name)

        engine_py = EngineAsynchronous(self._model)

        # ############################### Configure Jiminy #####################################

        robot_options = self._model.get_options()
        engine_options = engine_py.get_engine_options()
        ctrl_options = engine_py.get_controller_options()

        robot_options["telemetry"]["enableEncoderSensors"] = False
        engine_options["telemetry"]["enableConfiguration"] = False
        engine_options["telemetry"]["enableVelocity"] = False
        engine_options["telemetry"]["enableAcceleration"] = False
        engine_options["telemetry"]["enableTorque"] = False
        engine_options["telemetry"]["enableEnergy"] = False

        engine_options["stepper"]["solver"] = "runge_kutta_dopri5" # ["runge_kutta_dopri5", "explicit_euler"]

        self._model.set_options(robot_options)
        engine_py.set_engine_options(engine_options)
        engine_py.set_controller_options(ctrl_options)

        # ##################### Define some problem-specific variables #########################

        ## Torque magnitude of the action
        self.force_mag = 40.0

        ## Maximum absolute angle of the pole before considering the episode failed
        self.theta_threshold_radians = 25 * pi / 180
        ## Maximum absolute position of the cart before considering the episode failed
        self.x_threshold = 0.75

        # ####################### Configure the learning environment ###########################

        # The time step of the 'step' method
        dt = 2.0e-3

        super(JiminyCartPoleEnv, self).__init__("cartpole", engine_py, dt)

        # ##################### Define some problem-specific variables #########################

        # Bounds of the observation space.
        # Note that the Angle limit set to 2 * theta_threshold_radians
        # so failing observation is still within bounds
        high = np.array([self.x_threshold * 2,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float64).max,
                         np.finfo(np.float64).max])

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float64)

        self.state_random_high = np.array([0.5, 0.15, 0.1, 0.1])
        self.state_random_low = -self.state_random_high

        self.action_space = spaces.Discrete(2) # Force using a discrete action space


    def step(self, action):
        """
        @brief      Run a simulation step for a given.

        @param[in]  action  The action to perform (in the action space rather than
                            the original torque space).

        @return     The next state, the reward, the status of the simulation (done or not),
                    and an empty dictionary for compatibility with Gym OpenAI.
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Bypass 'action' setter and use direct assignment to max out the performances
        if action == 1:
            self.engine_py._action[0] = self.force_mag
        else:
            self.engine_py._action[0] = -self.force_mag
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
        x, theta, x_dot, theta_dot = self.state
        return        x < -self.x_threshold \
               or     x >  self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta >  self.theta_threshold_radians
