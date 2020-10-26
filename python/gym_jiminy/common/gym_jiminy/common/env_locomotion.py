import numpy as np
import numba as nb
from typing import Optional, Union, Tuple, Dict, Any

import gym

import jiminy_py.core as jiminy
from jiminy_py.core import (EncoderSensor as encoder,
                            EffortSensor as effort,
                            ContactSensor as contact,
                            ForceSensor as force,
                            ImuSensor as imu)
from jiminy_py.simulator import Simulator

import pinocchio as pin

from .env_bases import SpaceDictRecursive, BaseJiminyEnv
from .distributions import PeriodicGaussianProcess


MIN_GROUND_STIFFNESS_LOG = 5.5
MAX_GROUND_STIFFNESS_LOG = 7.0
MAX_GROUND_DAMPING_RATIO = 0.5
MIN_GROUND_FRICTION = 0.8
MAX_GROUND_FRICTION = 8.0

F_XY_IMPULSE_SCALE = 1000.0
F_XY_PROFILE_SCALE = 50.0
FLEX_STIFFNESS_SCALE = 1000
FLEX_DAMPING_SCALE = 10

SENSOR_DELAY_SCALE = {
    encoder.type: 3.0e-3,
    effort.type: 0.0,
    contact.type: 0.0,
    force.type: 0.0,
    imu.type: 0.0
}
SENSOR_NOISE_SCALE = {
    encoder.type:  np.array([0.0, 0.02]),
    effort.type: np.array([10.0]),
    contact.type: np.array([2.0, 2.0, 2.0, 10.0, 10.0, 10.0]),
    force.type: np.array([2.0, 2.0, 2.0]),
    imu.type:  np.array([0.0, 0.0, 0.0, 0.02, 0.02, 0.02, 0.2, 0.2, 0.2])
}

DEFAULT_SIMULATION_DURATION = 20.0  # (s) Default simulation duration
DEFAULT_ENGINE_DT = 1.0e-3  # (s) Stepper update period

DEFAULT_HLC_TO_LLC_RATIO = 1  # (NA)


class WalkerJiminyEnv(BaseJiminyEnv):
    """Implementation of a Gym environment for learning locomotion task for
    legged robots. It uses Jiminy Engine to perform physics evaluation and
    Meshcat for rendering.
    """

    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self,
                 urdf_path: str,
                 hardware_path: Optional[str] = None,
                 mesh_path: Optional[str] = None,
                 simu_duration_max: float = DEFAULT_SIMULATION_DURATION,
                 dt: float = DEFAULT_ENGINE_DT,
                 reward_mixture: Optional[dict] = None,
                 std_ratio: Optional[dict] = None,
                 config_path: Optional[str] = None,
                 avoid_instable_collisions: bool = True,
                 debug: bool = False,
                 **kwargs):
        """
        :param urdf_path: Path of the urdf model to be used for the simulation.
        :param hardware_path: Path of Jiminy hardware description toml file.
                              Optional: Looking for '.hdf' file in the same
                              folder and with the same name.
        :param mesh_path: Path to the folder containing the model meshes.
                          Optional: Env variable 'JIMINY_DATA_PATH' will be
                          used if available.
        :param simu_duration_max: Maximum duration of a simulation before
                                  returning done.
        :param dt: Engine timestep. It corresponds to the controller and
                   sensors update period.
        :param reward_mixture: Weighting factors of selected contributions to
                               total reward.
        :param std_ratio: Relative standard deviation of selected contributions
                          to environment stochasticity.
        :param config_path: Configuration toml file to import. It will be
                            imported AFTER loading the hardware description
                            file. It can be automatically generated from an
                            instance by calling `export_config_file` method.
                            Optional: Looking for '.config' file in the same
                            folder and with the same name. If not found,
                            using default configuration.
        :param avoid_instable_collisions: Prevent numerical instabilities by
                                          replacing collision mesh by vertices
                                          of associated minimal volume bounding
                                          box, and replacing primitive box by
                                          its vertices.
        :param debug: Whether or not the debug mode must be activated.
                      Doing it enables telemetry recording.
        :param kwargs: Keyword arguments to forward to `BaseJiminyEnv` class.
        """
        # Handling of default arguments
        if reward_mixture is None:
            reward_mixture = {
                'direction': 0.0,
                'energy': 0.0,
                'done': 1.0
            }
        reward_mixture = {k: v for k, v in reward_mixture.items() if v > 0.0}
        if std_ratio is None:
            std_ratio = {
                'model': 0.0,
                'ground': 0.0,
                'sensors': 0.0,
                'disturbance': 0.0,
            }
        std_ratio = {k: v for k, v in std_ratio.items() if v > 0.0}

        # Backup user arguments
        self.simu_duration_max = simu_duration_max
        self.reward_mixture = reward_mixture
        self.urdf_path = urdf_path
        self.mesh_path = mesh_path
        self.hardware_path = hardware_path
        self.config_path = config_path
        self.std_ratio = std_ratio
        self.avoid_instable_collisions = avoid_instable_collisions

        # Robot and engine internal buffers
        self._log_data = None
        self._forces_impulse = None
        self._forces_profile = None
        self._power_consumption_max = None

        # Configure and initialize the learning environment
        super().__init__(None, dt, debug, **kwargs)

    def _setup_environment(self) -> None:
        """Configure the environment.

        It is doing the following steps, successively:

            - creates a low-level engine is necessary,
            - updates some proxies that will be used for computing the
              reward and termination condition,
            - enforce some options of the low-level robot and engine,
            - randomize the environment according to 'std_ratio'.

        .. note::
            This method is called internally by `reset` method at the very
            beginning. This method can be overwritten to implement new
            contributions to the environment stochasticity, or to create
            custom low-level robot if the model must be different for each
            learning eposide for some reason.
        """
        # Check that a valid engine is available, and if not, create one
        if self.simulator is None:
            self.simulator = Simulator.build(
                self.urdf_path, self.hardware_path, self.mesh_path,
                has_freeflyer=True, use_theoretical_model=False,
                config_path=self.config_path,
                avoid_instable_collisions=self.avoid_instable_collisions,
                debug=self.debug)

        # Discard log data since no longer relevant
        self._log_data = None

        # Remove already register forces
        self._forces_impulse = []
        self._forces_profile = []
        self.simulator.remove_forces()

        # Update some internal buffers used for computing the reward
        motor_effort_limit = self.robot.effort_limit[
            self.robot.motors_velocity_idx]
        motor_velocity_limit = self.robot.velocity_limit[
            self.robot.motors_velocity_idx]
        self._power_consumption_max = sum(
            motor_effort_limit * motor_velocity_limit)

        # Compute the height of the freeflyer in neutral configuration
        # TODO: Take into account the ground profile.
        if self.robot.has_freeflyer:
            q0, _ = self._sample_state()
            self._height_neutral = q0[2]
        else:
            self._height_neutral = None

        # Get the options of robot and engine
        robot_options = self.robot.get_options()
        engine_options = self.simulator.engine.get_options()

        # Make sure to log at least required data for reward
        # computation and log replay
        engine_options['telemetry']['enableConfiguration'] = True

        # Enable the flexible model
        robot_options["model"]["dynamics"]["enableFlexibleModel"] = True

        # Set maximum number of iterations by seconds in average
        engine_options["stepper"]["iterMax"] = \
            int(self.simu_duration_max / 1.0e-4)
        # Set maximum computation time for single internal integration steps
        engine_options["stepper"]["timeout"] = 1.0

        # ============= Add some stochasticity to the environment =============

        # Change ground friction and sprint-dumper contact dynamics
        engine_options['contacts']['stiffness'] = 10 ** (
            (MAX_GROUND_STIFFNESS_LOG - MIN_GROUND_STIFFNESS_LOG) / 2 *
            self.std_ratio.get('ground', 0.0) *
            self.rg.uniform(low=-1.0, high=1.0) +
            (MAX_GROUND_STIFFNESS_LOG + MIN_GROUND_STIFFNESS_LOG) / 2)
        engine_options['contacts']['damping'] = (
            self.rg.uniform(MAX_GROUND_DAMPING_RATIO) *
            2.0 * np.sqrt(engine_options['contacts']['stiffness']))
        engine_options['contacts']['frictionDry'] = (
            (MAX_GROUND_FRICTION - MIN_GROUND_FRICTION) *
            self.std_ratio.get('ground', 0.0) * self.rg.uniform() +
            MIN_GROUND_FRICTION)
        engine_options['contacts']['frictionViscous'] = \
            engine_options['contacts']['frictionDry']

        # Add sensor noise, bias and delay
        if 'sensors' in self.std_ratio.keys():
            for sensor in (encoder, effort, contact, force, imu):
                sensors_options = robot_options["sensors"][sensor.type]
                for sensor_options in sensors_options.values():
                    sensor_options['delay'] = self.std_ratio['sensors'] * \
                        self.rg.uniform() * SENSOR_DELAY_SCALE[sensor.type]
                    sensor_options['noiseStd'] = self.std_ratio['sensors'] * \
                        self.rg.uniform() * SENSOR_NOISE_SCALE[sensor.type]

        # Randomize the flexibility parameters
        if 'model' in self.std_ratio.keys():
            dynamics_options = robot_options["model"]["dynamics"]
            for flexibility in dynamics_options["flexibilityConfig"]:
                flexibility['stiffness'] += self.std_ratio['model'] * \
                    FLEX_STIFFNESS_SCALE * self.rg.uniform(low=-1.0, high=1.0)
                flexibility['damping'] += self.std_ratio['model'] * \
                    FLEX_DAMPING_SCALE * self.rg.uniform(low=-1.0, high=1.0)

        # Apply the disturbance to the first actual body
        if 'disturbance' in self.std_ratio.keys():
            # Determine the actual root body of the kinematic tree
            for frame in self.robot.pinocchio_model.frames:
                if frame.type == pin.FrameType.BODY and frame.parent == 1:
                    break
            frame_name = frame.name

            # Schedule some external impulse forces applied on PelvisLink
            for t_ref in np.arange(0.0, self.simu_duration_max, 2.0)[1:]:
                F_xy_impulse = self.rg.randn(2)
                F_xy_impulse /= np.linalg.norm(F_xy_impulse, ord=2)
                F_xy_impulse *= self.std_ratio['disturbance'] * \
                    F_XY_IMPULSE_SCALE * self.rg.uniform()
                F = np.concatenate((F_xy_impulse, np.zeros(4)))
                t = t_ref + self.rg.uniform(low=-1.0, high=1.0) * 250e-3
                force_impulse = {
                    'frame_name': frame_name,
                    't': t, 'dt': 10e-3, 'F': F
                }
                self.simulator.register_force_impulse(**force_impulse)
                self._forces_impulse.append(force_impulse)

            # Schedule a single force profile applied on PelvisLink.
            # Internally, it relies on a linear interpolation instead
            # of a spline for the sake of efficiency, since accuracy
            # is not a big deal, and the derivative is not needed.
            n_timesteps = 50
            t_profile = np.linspace(0.0, 1.0, n_timesteps + 1)
            F_xy_profile = PeriodicGaussianProcess(
                mean=np.zeros((2, n_timesteps + 1)),
                scale=(self.std_ratio['disturbance'] *
                       F_XY_PROFILE_SCALE * np.ones(2)),
                wavelength=np.tensor([1.0, 1.0]),
                period=np.tensor([1.0]),
                dt=np.tensor([1 / n_timesteps])
            ).sample().T

            @nb.jit(nopython=True, nogil=True)
            def F_xy_profile_interp1d(t):
                t_rel = t % 1.0
                t_ind = np.searchsorted(t_profile, t_rel, 'right') - 1
                ratio = (t_rel - t_profile[t_ind]) * n_timesteps
                return (1 - ratio) * F_xy_profile[t_ind] + \
                    ratio * F_xy_profile[t_ind + 1]

            F_xy_profile_interp1d(0)  # Pre-compilation
            self.F_xy_profile_spline = F_xy_profile_interp1d
            force_profile = {
                'frame_name': 'PelvisLink',
                'force_function': self._force_external_profile
            }
            self.simulator.register_force_profile(**force_profile)
            self._forces_profile.append(force_profile)

        # Set the options, finally
        self.robot.set_options(robot_options)
        self.simulator.engine.set_options(engine_options)

    def _get_time_space(self) -> None:
        """Get time space.

        It takes advantage of knowing the maximum simulation duration to shrink
        down the range. Note that observation will be out-of-bounds steps are
        performed after this point.
        """
        return gym.spaces.Box(
            low=0.0, high=self.simu_duration_max, shape=(1,),
            dtype=np.float64)

    def _force_external_profile(self,
                                t: float,
                                q: np.ndarray,
                                v: np.ndarray,
                                F: np.ndarray) -> None:
        """User-specified pre- or post- processing of the external force
        profile.

        Typical usecases are time rescaling (1.0 second by default), or
        changing the orientation of the force (x/y in world frame by default).
        It could also be used for clamping the force.
        """
        t_scaled = t / (2 * self.gait_features["step_length"])
        F[:2] = self.F_xy_profile_spline(t_scaled)

    def _is_done(self) -> bool:
        """Determine whether the episode is over.

        The termination conditions are the following:

            - fall detection (enabled if the robot has a freeflyer):
              the freeflyer goes lower than 75% of its height in
              neutral configuration.
            - maximum simulation duration exceeded
        """
        if self.robot.has_freeflyer:
            if self._state[0][2] < self._height_neutral * 0.75:
                return True
        if self.simulator.stepper_state.t >= self.simu_duration_max:
            return True
        return False

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """Compute reward at current episode state.

        It computes the reward associated with each individual contribution
        according to 'reward_mixture'.

        .. note::
            This method can be overwritten to implement new contributions to
            the reward, or to monitor more information.

        :returns: [0] Total reward.
                  [1] Value of each contribution as a dictionary.
        """
        reward_dict = {}

        # Define some proxies
        reward_mixture_keys = self.reward_mixture.keys()

        if 'energy' in reward_mixture_keys:
            v_mot = self.robot.sensors_data[encoder.type][1]
            power_consumption = sum(np.maximum(self.action_prev * v_mot, 0.0))
            power_consumption_rel = \
                power_consumption / self._power_consumption_max
            reward_dict['energy'] = - power_consumption_rel

        if 'done' in reward_mixture_keys:
            reward_dict['done'] = 1.0

        # Compute the total reward
        reward_total = sum([self.reward_mixture[name] * value
                            for name, value in reward_dict.items()])

        return reward_total, reward_dict

    def _compute_reward_terminal(self):
        """Compute the reward at the end of the episode.

        It computes the terminal reward associated with each individual
        contribution according to 'reward_mixture'.
        """
        reward_dict = {}

        reward_mixture_keys = self.reward_mixture.keys()

        # Add a negative reward proportional to the average deviation on
        # Y-axis. It is equal to 0.0 if the frontal displacement is perfectly
        # symmetric wrt Y-axis over the whole trajectory.
        if 'direction' in reward_mixture_keys:
            if self.robot.has_freeflyer:
                frontal_displacement = abs(np.mean(self._log_data[
                    'HighLevelController.currentFreeflyerPositionTransY']))
            else:
                frontal_displacement = 0.0
            reward_dict['direction'] = - frontal_displacement

        # Compute the total reward
        reward_total = sum([self.reward_mixture[name] * value
                            for name, value in reward_dict.items()])

        return reward_total, reward_dict


class WalkerPDControlJiminyEnv(WalkerJiminyEnv):
    def __init__(self,
                 urdf_path: str,
                 hardware_path: Optional[str] = None,
                 mesh_path: Optional[str] = None,
                 simu_duration_max: float = DEFAULT_SIMULATION_DURATION,
                 dt: float = DEFAULT_ENGINE_DT,
                 hlc_to_llc_ratio: int = DEFAULT_HLC_TO_LLC_RATIO,
                 pid_kp: Union[float, np.ndarray] = 0.0,
                 pid_kd: Union[float, np.ndarray] = 0.0,
                 reward_mixture: Optional[dict] = None,
                 std_ratio: Optional[dict] = None,
                 config_path: Optional[str] = None,
                 avoid_instable_collisions: bool = True,
                 debug: bool = False):
        """
        :param urdf_path: Path of the urdf model to be used for the simulation.
        :param hardware_path: Path of Jiminy hardware description toml file.
                              Optional: Looking for '.hdf' file in the same
                              folder and with the same name.
        :param mesh_path: Path to the folder containing the model meshes.
                          Optional: Env variable 'JIMINY_DATA_PATH' will be
                          used if available.
        :param simu_duration_max: Maximum duration of a simulation before
                                  returning done.
        :param dt: Engine timestep. It corresponds to the controller and
                   sensors update period.
        :param hlc_to_llc_ratio: High-level to Low-level control frequency
                                 ratio. More precisely, at each step, the
                                 command torque is: updated 'hlc_to_llc_ratio'
                                 times while the target motor state is only
                                 updated once.
        :param pid_kp: PD controller position-proportional gain in motor order.
        :param pid_kd: PD controller velocity-proportional gain in motor order.
        :param reward_mixture: Weighting factors of selected contributions to
                               total reward.
        :param std_ratio: Relative standard deviation of selected contributions
                          to environment stochasticity.
        :param config_path: Configuration toml file to import. It will be
                            imported AFTER loading the hardware description
                            file. It can be automatically generated from an
                            instance by calling `export_config_file` method.
                            Optional: Looking for '.config' file in the same
                            folder and with the same name. If not found,
                            using default configuration.
        :param avoid_instable_collisions: Prevent numerical instabilities by
                                          replacing collision mesh by vertices
                                          of associated minimal volume bounding
                                          box, and replacing primitive box by
                                          its vertices.
        :param debug: Whether or not the debug mode must be activated.
                      Doing it enables telemetry recording.
        """
        # Backup some user arguments
        self.hlc_to_llc_ratio = hlc_to_llc_ratio
        self.pid_kp = pid_kp
        self.pid_kd = pid_kd

        # Low-level controller buffers
        self.motor_to_encoder = None
        self._q_target = None
        self._v_target = None

        # Initialize the environment
        super().__init__(
            urdf_path, hardware_path, mesh_path, simu_duration_max, dt,
            reward_mixture, std_ratio, config_path, avoid_instable_collisions,
            debug)

    def _setup_environment(self) -> None:
        """Configure the environment.

        In addition of doing the same than the base implementation, it also
        updates the mapping from motors to encoders indices.
        """
        # Setup the environment as usual
        super()._setup_environment()

        # Refresh the mapping between the motors and encoders
        encoder_joints = []
        for name in self.robot.sensors_names[encoder.type]:
            sensor = self.robot.get_sensor(encoder.type, name)
            encoder_joints.append(sensor.joint_name)

        self.motor_to_encoder = []
        for name in self.robot.motors_names:
            motor = self.robot.get_motor(name)
            motor_joint = motor.joint_name
            encoder_found = False
            for i, encoder_joint in enumerate(encoder_joints):
                if motor_joint == encoder_joint:
                    self.motor_to_encoder.append(i)
                    encoder_found = True
                    break
            if not encoder_found:
                raise RuntimeError(
                    "No encoder sensor associated with motor '{name}'. Every "
                    "actuated joint must have an encoder sensor attached.")

    def _refresh_action_space(self) -> None:
        """Configure the action space of the environment.

        The action spaces corresponds to the position and velocity of motors
        instead of the torque, as it is the case by default.
        """
        # Extract the position and velocity bounds for the observation space
        encoder_space = self._get_sensors_space()[encoder.type]
        pos_high, vel_high = encoder_space.high
        pos_low, vel_low = encoder_space.low

        # Reorder the position and velocity bounds to match motors order
        pos_high = pos_high[self.motor_to_encoder]
        pos_low = pos_low[self.motor_to_encoder]
        vel_high = vel_high[self.motor_to_encoder]
        vel_low = vel_low[self.motor_to_encoder]

        # Set the action space. Note that it is flattened.
        self.action_space = gym.spaces.Box(
            low=np.concatenate((pos_low, vel_low)),
            high=np.concatenate((pos_high, vel_high)),
            dtype=np.float64)

    def _send_command(self,
                      t: float,
                      q: np.ndarray,
                      v: np.ndarray,
                      sensors_data: jiminy.sensorsData,
                      u_command: np.ndarray) -> None:
        """Compute the motor torques using a PD controller.

        It is based on the error between the measured motors positions and
        velocities and the desired one.
        """
        # Compute command if the simulation is running, otherwise do nothing
        if self.simulator.is_simulation_running:
            # Estimate position and motor velocity from encoder data
            q_enc, v_enc = sensors_data[encoder.type][:, self.motor_to_encoder]

            # Compute the joint tracking error
            q_err = q_enc - self._q_target
            v_err = v_enc - self._v_target

            # Compute PD command
            u_command[:] = - self.pid_kp * (q_err + self.pid_kd * v_err)
        else:
            u_command[:] = 0.0

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Reset the simulation and specify the initial state of the robot.

        It is the same that the base implementation, except that it also reset
        the internal state of the PD controller.
        """
        self._q_target = qpos[sum(self.robot.motors_position_idx, [])]
        self._v_target = qvel[self.robot.motors_velocity_idx]
        super().set_state(qpos, qvel)

    def step(self,
             action: Optional[np.ndarray] = None
             ) -> Tuple[SpaceDictRecursive, float, bool, Dict[str, Any]]:
        """Run a simulation step for a given action.

        :param action: Flattened array gathering the target motors positions
                       and velocities in this order.

        :returns: The next observation, the reward, the status of the episode
                  (done or not), and a dictionary of extra information
        """
        # Update target motor state
        self._q_target, self._v_target = np.split(action, 2, axis=-1)

        # Run the whole simulation in one go
        reward = 0.0
        reward_info = {k: [] for k in self.reward_mixture.keys()}
        for _ in range(self.hlc_to_llc_ratio):
            obs, reward_step, done, info_step = super().step(action)
            for k, v in info_step.get('reward', {}).items():
                reward_info[k].append(v)
            reward += reward_step
            if done:
                break

        # Extract additional information
        info = {'reward': reward_info, 't_end': self.simulator.stepper_state.t}

        return obs, reward, done, info
