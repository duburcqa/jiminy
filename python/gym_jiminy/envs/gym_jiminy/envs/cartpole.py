""" TODO: Write documentation.
"""
import os
import numpy as np
from typing import Dict, Any, Optional, Tuple

from gymnasium import spaces

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator

from gym_jiminy.common.utils import sample, DataNested
from gym_jiminy.common.bases import InfoType
from gym_jiminy.common.envs import EngineObsType, BaseJiminyEnv

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


# Stepper update period
STEP_DT = 0.02
# Controller update period
CONTROL_DT = 0.02
# Maximum absolute position of the cart before considering the episode failed
X_THRESHOLD = 2.4
# Maximum absolute angle of the pole before considering the episode failed
THETA_THRESHOLD = 12.0 * np.pi / 180.0
# Sampling range for cart position
X_RANDOM_MAX = 0.05
# Sampling range for pole angle
THETA_RANDOM_MAX = 0.05
# Sampling range for cart linear velocity
DX_RANDOM_MAX = 0.05
# Sampling range for pole angular velocity
DTHETA_RANDOM_MAX = 0.05


class CartPoleJiminyEnv(BaseJiminyEnv[np.ndarray, np.ndarray]):
    """Implementation of a Gym environment for the Cartpole which is using
    Jiminy Engine to perform physics computations and Meshcat for rendering.

    It is a specialization of BaseJiminyEnv. The Cartpole is a pole attached
    by an un-actuated joint to a cart. The goal is to prevent the pendulum from
    falling over by increasing and reducing the cart's velocity.

    **OBSERVATION:**
    Type: Box(4)
    Num	Observation              Min         Max
    0	Cart Position           -4.8         4.8
    1	Cart Velocity           -Inf         Inf
    2	Pole Angle              -24 deg      24 deg
    3	Pole Velocity At Tip    -Inf         Inf

    **ACTIONS:**
    Type: Discrete(2)
    Num	Action
    0	Push cart to the left
    1	Push cart to the right

    Note that the amount the velocity that is reduced or increased is not
    fixed, it depends on the angle the pole is pointing. This is because the
    center of gravity of the pole increases the amount of energy needed to move
    the cart underneath it.

    **REWARD:**
    Reward is 1 for every step taken, including the termination step.

    **STARTING STATE:**
    All observations are assigned a uniform random value in range [-0.05, 0.05]

    **EPISODE TERMINATION:**
    If any of these conditions is satisfied:

        - Pole Angle is more than 12 degrees Cart
        - Position is more than 2.4
        - Episode length is greater than 200

    **SOLVED REQUIREMENTS:**
    Considered solved when the average reward is greater than or equal to 195.0
    over 100 consecutive trials.
    """
    def __init__(self,
                 continuous: bool = False,
                 debug: bool = False,
                 viewer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        :param continuous: Whether the action space is continuous. If not
                           continuous, the action space has only 3 states, i.e.
                           low, zero, and high.
                           Optional: True by default.
        :param debug: Whether the debug mode must be enabled.
                      See `BaseJiminyEnv` constructor for details.
        :param viewer_kwargs: Keyword arguments to override by default whenever
                              a viewer must be instantiated
                              See `Simulator` constructor for details.
        """
        # Backup some input arguments
        self.continuous = continuous

        # Get URDF path
        data_dir = str(files("gym_jiminy.envs") / "data/toys_models/cartpole")
        urdf_path = os.path.join(data_dir, "cartpole.urdf")

        # Instantiate robot
        robot = jiminy.Robot()
        robot.initialize(
            urdf_path, has_freeflyer=False, mesh_package_dirs=[data_dir])

        # Add motors and sensors
        motor_joint_name = "slider_to_cart"
        encoder_sensors_descr = {
            "slider": "slider_to_cart",
            "pole": "cart_to_pole"
        }
        motor = jiminy.SimpleMotor(motor_joint_name)
        robot.attach_motor(motor)
        motor.initialize(motor_joint_name)
        for sensor_name, joint_name in encoder_sensors_descr.items():
            encoder = jiminy.EncoderSensor(sensor_name)
            robot.attach_sensor(encoder)
            encoder.initialize(joint_name)

        # Instantiate simulator
        simulator = Simulator(robot)

        # OpenAI Gym implementation of Cartpole has no velocity limit
        model_options = simulator.robot.get_model_options()
        model_options["joints"]["enableVelocityLimit"] = False
        simulator.robot.set_model_options(model_options)

        # Map between discrete actions and actual motor force if necessary
        if not self.continuous:
            self.AVAIL_CTRL = [-motor.command_limit, motor.command_limit]

        # Configure the learning environment
        super().__init__(simulator, step_dt=STEP_DT, debug=debug)

        # Create some proxies for fast access
        self.__state_view = (
            self._observation[:self.robot.nq],
            self._observation[self.robot.nq:(self.robot.nq+self.robot.nv)])

    def _setup(self) -> None:
        """ TODO: Write documentation.
        """
        # Call base implementation
        super()._setup()

        # OpenAI Gym implementation uses euler explicit integration scheme
        engine_options = self.simulator.engine.get_options()
        engine_options["stepper"]["solver"] = "euler_explicit"
        engine_options["stepper"]["dtMax"] = CONTROL_DT
        self.simulator.engine.set_options(engine_options)

    def _initialize_observation_space(self) -> None:
        """Configure the observation of the environment.

        Implement the official Gym cartpole-v1 action space. Only the state is
        observable, while by default, the current time, state, and sensors data
        are available.

        The Angle limit set to 2 times the failure thresholds, so that
        observations of failure are still within bounds.

        See documentation: https://gym.openai.com/envs/CartPole-v1/.
        """
        # Compute observation bounds
        high = np.array([2.0 * X_THRESHOLD,
                         2.0 * THETA_THRESHOLD,
                         *self.robot.velocity_limit])

        # Set the observation space
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float64)

    def _initialize_action_space(self) -> None:
        """ TODO: Write documentation.

        Replace the action space by its discrete representation depending on
        'continuous'.
        """
        if not self.continuous:
            self.action_space = spaces.Discrete(len(self.AVAIL_CTRL))
        else:
            super()._initialize_action_space()

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """ TODO: Write documentation.

        Bounds of hypercube associated with initial state of robot.
        """
        qpos = sample(scale=np.array([
            X_RANDOM_MAX, THETA_RANDOM_MAX]), rg=self.np_random)
        qvel = sample(scale=np.array([
            DX_RANDOM_MAX, DTHETA_RANDOM_MAX]), rg=self.np_random)
        return qpos, qvel

    def refresh_observation(self, measurement: EngineObsType) -> None:
        self.__state_view[0][:] = measurement['agent_state']['q']
        self.__state_view[1][:] = measurement['agent_state']['v']

    def has_terminated(self) -> Tuple[bool, bool]:
        """ TODO: Write documentation.
        """
        x, theta, *_ = self._observation
        done = (abs(x) > X_THRESHOLD) or (abs(theta) > THETA_THRESHOLD)
        return done, False

    def compute_command(self,
                        observation: DataNested,
                        action: np.ndarray
                        ) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        Convert a discrete action into its actual value if necessary.

        :param observation: Observation of the environment.
        :param action: Desired motors efforts.
        """
        # Call base implementation
        action = super().compute_command(observation, action)

        # Compute the actual torque to apply
        if not self.continuous:
            action = self.AVAIL_CTRL[round(action[()])]

        return action

    def compute_reward(self,
                       done: bool,
                       truncated: bool,
                       info: InfoType) -> float:
        """ TODO: Write documentation.

        Add a small positive reward as long as a terminal condition has
        never been reached during the current episode.
        """
        return 1.0 if not done else 0.0

    def _key_to_action(self, key: Optional[str], **kwargs: Any) -> np.ndarray:
        """ TODO: Write documentation.
        """
        if key is None:
            return None
        if key == "Left":
            return 1
        if key == "Right":
            return 0
        print(f"Key '{key}' not bound to any action.")
        return None
