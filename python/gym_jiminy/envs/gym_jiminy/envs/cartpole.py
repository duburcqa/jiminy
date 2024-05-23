""" TODO: Write documentation.
"""
import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
from gymnasium import spaces

import jiminy_py.core as jiminy
from jiminy_py.core import array_copyto  # pylint: disable=no-name-in-module
from jiminy_py.simulator import Simulator

from gym_jiminy.common.bases import InfoType, EngineObsType
from gym_jiminy.common.envs import BaseJiminyEnv
from gym_jiminy.common.utils import sample

if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files


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
        :param viewer_kwargs: Keyword arguments used to override the original
                              default values whenever a viewer is instantiated.
                              This is the only way to pass custom arguments to
                              the viewer when calling `render` method, unlike
                              `replay` which forwards extra keyword arguments.
                              Optional: None by default.
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
        motor = jiminy.SimpleMotor(motor_joint_name)
        robot.attach_motor(motor)
        motor.initialize(motor_joint_name)
        for sensor_name, joint_name in (
                ("slider", "slider_to_cart"), ("pole", "cart_to_pole")):
            encoder = jiminy.EncoderSensor(sensor_name)
            robot.attach_sensor(encoder)
            encoder.initialize(joint_name=joint_name)

        # Instantiate simulator
        simulator = Simulator(robot, viewer_kwargs=viewer_kwargs)

        # OpenAI Gym implementation of Cartpole has no velocity limit
        motor_options = motor.get_options()
        motor_options["enableVelocityLimit"] = False
        motor.set_options(motor_options)

        # Map between discrete actions and actual motor force if necessary
        if not self.continuous:
            command_limit = np.array(motor.effort_limit)
            self.AVAIL_CTRL = (-command_limit, np.array(0.0), command_limit)

        # Configure the learning environment
        super().__init__(simulator, step_dt=STEP_DT, debug=debug)

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # OpenAI Gym implementation uses euler explicit integration scheme
        engine_options = self.simulator.get_options()
        engine_options["stepper"]["odeSolver"] = "euler_explicit"
        engine_options["stepper"]["dtMax"] = CONTROL_DT
        self.simulator.set_options(engine_options)

    def _initialize_observation_space(self) -> None:
        """Configure the observation of the environment.

        Implement the official Gym cartpole-v1 observation space. Only the
        theoretical state of the pendulum is observed, namely the position and
        velocity of the cart plus the pole angle and its angular velocity.

        See documentation: https://gym.openai.com/envs/CartPole-v1/.
        """
        # Define custom symmetric position bounds
        position_limit_upper = np.array([X_THRESHOLD, THETA_THRESHOLD])

        # Get velocity bounds associated the theoretical model
        velocity_limit = self.robot.get_theoretical_velocity_from_extended(
            self.robot.pinocchio_model.velocityLimit)

        # Set the observation space
        state_limit_upper = np.concatenate((
            position_limit_upper, velocity_limit))
        self.observation_space = spaces.Box(
            low=-state_limit_upper, high=state_limit_upper, dtype=np.float64)

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
        x, theta = sample(scale=np.array(
            [X_RANDOM_MAX, THETA_RANDOM_MAX]), rg=self.np_random)
        qpos = np.array([x, np.cos(theta), np.sin(theta)])
        qvel = sample(scale=np.array([
            DX_RANDOM_MAX, DTHETA_RANDOM_MAX]), rg=self.np_random)
        return qpos, qvel

    def refresh_observation(self, measurement: EngineObsType) -> None:
        obs = measurement['measurements']['EncoderSensor']
        array_copyto(self.observation, obs.reshape((-1,)))

    def compute_command(self, action: np.ndarray, command: np.ndarray) -> None:
        """Compute the motors efforts to apply on the robot.

        Convert a discrete action into its actual value if necessary.

        :param action: Desired motors efforts.
        """
        command[:] = action if self.continuous else self.AVAIL_CTRL[action]

    def compute_reward(self, terminated: bool, info: InfoType) -> float:
        """Compute reward at current episode state.

        Constant positive reward equal to 1.0 as long as no termination
        condition has been triggered.
        """
        return 1.0 if not terminated else 0.0

    def _key_to_action(self,
                       key: str,
                       obs: np.ndarray,
                       reward: Optional[float],
                       **kwargs: Any) -> Optional[np.ndarray]:
        if key == "Left":
            return np.array(1)
        if key == "Right":
            return np.array(0)
        logging.warning(f"Key '{key}' not bound to any action.")
        return None
