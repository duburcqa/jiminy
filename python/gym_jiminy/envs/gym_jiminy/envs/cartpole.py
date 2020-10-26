import os
import numpy as np
from pkg_resources import resource_filename
from typing import Optional, Tuple, Dict, Any

from gym import spaces

from jiminy_py import core as jiminy
from jiminy_py.simulator import Simulator

from ..common.env_bases import SpaceDictRecursive, BaseJiminyEnv


# Stepper update period
STEP_DT = 0.02
# Maximum absolute position of the cart before considering the episode failed
X_THRESHOLD = 2.4
# Maximum absolute angle of the pole before considering the episode failed
THETA_THRESHOLD = 12.0 * np.pi / 180.0
# Sampling range for cart position
X_RANDOM_RANGE = 0.05
# Sampling range for pole angle
THETA_RANDOM_RANGE = 0.05
# Sampling range for cart linear velocity
DX_RANDOM_RANGE = 0.05
# Sampling range for pole angular velocity
DTHETA_RANDOM_RANGE = 0.05


class CartPoleJiminyEnv(BaseJiminyEnv):
    """
    @brief      Implementation of a Gym environment for the Cartpole which is
                using Jiminy Engine to perform physics computations and Meshcat
                for rendering.

    @remark     It is a specialization of BaseJiminyEnv. The Cartpole is a pole
                attached by an un-actuated joint to a cart. The goal is to
                prevent the pendulum from falling over by increasing and
                reducing the cart's velocity.

    @details    **OBSERVATION:**
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

                Note that the amount the velocity that is reduced or increased
                is not fixed, it depends on the angle the pole is pointing.
                This is because the center of gravity of the pole increases the
                amount of energy needed to move the cart underneath it.

                **REWARD:**
                Reward is 1 for every step taken, including the termination
                step.

                **STARTING STATE:**
                All observations are assigned a uniform random value in range
                [-0.05, 0.05]

                **EPISODE TERMINATION:**
                If any of these conditions is satisfied:
                    - Pole Angle is more than 12 degrees Cart
                    - Position is more than 2.4
                    - Episode length is greater than 200

                **SOLVED REQUIREMENTS:**
                Considered solved when the average reward is greater than or
                equal to 195.0 over 100 consecutive trials.
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

        # Get URDF path
        data_dir = resource_filename('gym_jiminy.envs', 'data/toys_models')
        urdf_path = os.path.join(data_dir, "cartpole/cartpole.urdf")

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

        # Map between discrete actions and actual motor force if necessary
        if not self.continuous:
            self.AVAIL_FORCE = [-motor.effort_limit, motor.effort_limit]

        # Bounds of hypercube associated with initial state of robot
        self.position_random_range = np.array([
            THETA_RANDOM_RANGE, X_RANDOM_RANGE])
        self.velocity_random_range = np.array([
            DX_RANDOM_RANGE, DTHETA_RANDOM_RANGE])

        # Configure the learning environment
        super().__init__(simulator, STEP_DT, debug=False)

    def _setup_environment(self) -> None:
        """
        @brief    TODO
        """
        super()._setup_environment()

        # OpenAI Gym implementation of Cartpole has no velocity limit
        robot_options = self.robot.get_options()
        robot_options["model"]["joints"]["enableVelocityLimit"] = False
        self.robot.set_options(robot_options)

    def _refresh_observation_space(self) -> None:
        """
        @brief Configure the observation of the environment.

        @details Implement the official Gym cartpole-v1 action space. Only the
                 state is observable, while by default, the current time,
                 state, and sensors data are available.See documentation
                 https://gym.openai.com/envs/CartPole-v1/.

        @remark The Angle limit set to 2 times the failure thresholds, so that
                observations of failure are still within bounds.
        """
        # Compute observation bounds
        high = np.array([2.0 * X_THRESHOLD,
                         2.0 * THETA_THRESHOLD,
                         *self.robot.velocity_limit])

        # Set the observation space
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float64)

    def _fetch_obs(self) -> None:
        # @copydoc BaseJiminyEnv::_fetch_obs
        return np.concatenate(self._state)

    def _refresh_action_space(self) -> None:
        """
        @brief    TODO

        @details Replace the action space by its discrete representation
                 depending on 'continuous'.
        """
        if not self.continuous:
            self.action_space = spaces.Discrete(len(self.AVAIL_FORCE))
        else:
            super()._refresh_action_space()

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        @brief    TODO
        """
        qpos = self.rg.uniform(low=-self.position_random_range,
                               high=self.position_random_range)
        qvel = self.rg.uniform(low=-self.velocity_random_range,
                               high=self.velocity_random_range)
        return qpos, qvel

    @staticmethod
    def _key_to_action(key: str) -> np.ndarray:
        """
        @brief    TODO
        """
        if key == "Left":
            return 1
        elif key == "Right":
            return 0
        else:
            print(f"Key {key} is not bound to any action.")
            return None

    def _is_done(self) -> bool:
        """
        @brief    TODO
        """
        x, theta, _, _ = self.get_obs()
        return (abs(x) > X_THRESHOLD) or (abs(theta) > THETA_THRESHOLD)

    def _compute_reward(self) -> Tuple[float, Dict[str, Any]]:
        """
        @brief    TODO

        @details Add a small positive reward as long as the termination
                 condition has never been reached during the same episode.
        """
        reward = 0.0
        if not self._num_steps_beyond_done:  # True for both None and 0
            reward += 1.0
        return reward, {}

    def step(self,
             action: Optional[np.ndarray] = None
             ) -> Tuple[SpaceDictRecursive, float, bool, Dict[str, Any]]:
        """
        @brief    TODO
        """
        # @copydoc BaseJiminyEnv::step
        # Compute the actual force to apply
        if not self.continuous and action is not None:
            action = self.AVAIL_FORCE[action]

        # Perform the step
        return super().step(action)

    def render(self, mode: str = 'human', **kwargs) -> Optional[np.ndarray]:
        if not self.simulator._is_viewer_available:
            kwargs["camera_xyzrpy"] = [(0.0, 7.0, 0.0), (np.pi/2, 0.0, np.pi)]
        return super().render(mode, **kwargs)
