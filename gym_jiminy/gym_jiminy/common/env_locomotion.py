## @file

import numpy as np
import numba as nb
from typing import Optional

from jiminy_py.core import (EncoderSensor as encoder,
                            EffortSensor as effort,
                            ContactSensor as contact,
                            ForceSensor as force,
                            ImuSensor as imu)
from jiminy_py.engine import BaseJiminyEngine
from jiminy_py.dynamics import compute_freeflyer_state_from_fixed_body

import pinocchio as pin
from pinocchio import neutral

from .env_bases import BaseJiminyEnv
from .distributions import PeriodicGaussianProcess


MIN_GROUND_STIFFNESS_LOG = 5.5
MAX_GROUND_STIFFNESS_LOG = 7.5
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

DEFAULT_HLC_TO_LLC_RATIO = 1 # (NA)
PID_KP = 20000.0
PID_KD = 0.01


class WalkerTorqueControlJiminyEnv(BaseJiminyEnv):
    """
    @brief Implementation of a Gym environment for learning locomotion task for
           legged robots. It uses Jiminy Engine to perform physics evaluation
           and Meshcat for rendering.
    """

    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self,
                 urdf_path: str,
                 toml_path: Optional[str] = None,
                 mesh_path: Optional[str] = None,
                 simu_duration_max: float = DEFAULT_SIMULATION_DURATION,
                 dt: float = DEFAULT_ENGINE_DT,
                 reward_mixture: Optional[dict] = None,
                 std_ratio: Optional[dict] = None,
                 debug: bool = False):
        """
        @brief Constructor.

        @param simu_duration_max  Maximum duration of a simulation before
                                  returning done.
        @param dt  Engine timestep. It corresponds to the controller and
                   sensors update period.
        @param urdf_path  Path of the urdf model to be used for the simulation.
        @param toml_path  Path of the Jiminy hardware description file.
                          Optional: Looking for toml file in the same folder
                          and with the same name.
        @param mesh_path  Path to the folder containing the model meshes.
                          Optional: Env variable 'JIMINY_DATA_PATH' will be
                          used if available.
        @param reward_mixture  Weighting factors of selected contributions to
                               total reward.
        @param std_ratio  Relative standard deviation of selected contributions
                          to environment stochasticity.
        @param debug  Whether or not the debug mode must be activated.
                      Doing it enables telemetry recording.
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
        self.toml_path = toml_path
        self.mesh_path = mesh_path
        self.std_ratio = std_ratio

        # Robot and engine internal buffers
        self._log_data = None
        self._forces_impulse = None
        self._forces_profile = None
        self._power_consumption_max = None

        # Configure and initialize the learning environment
        super().__init__(None, dt, debug)

    def _setup_environment(self):
        """
        @brief    TODO
        """
        # Make sure a valid engine is available
        if self.engine_py is None:
            # Instantiate a new engine
            self.engine_py = BaseJiminyEngine(
                self.urdf_path, self.toml_path, self.mesh_path,
                has_freeflyer=True, use_theoretical_model=False,
                debug=self.debug)

        # Discard log data since no longer relevant
        self._log_data = None

        # Remove already register forces
        self._forces_impulse = []
        self._forces_profile = []
        self.engine.remove_forces()

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
            q0 = neutral(self.robot.pinocchio_model)
            compute_freeflyer_state_from_fixed_body(self.robot, q0,
                ground_profile=None, use_theoretical_model=False)
            self._height_neutral = q0[2]
        else:
            self._height_neutral = None

        # Override some options of the robot and engine
        robot_options = self.robot.get_options()
        engine_options = self.engine.get_options()

        ### Make sure to log at least required data for reward
        #   computation and log replay
        engine_options['telemetry']['enableConfiguration'] = True

        ### Enable the flexible model
        robot_options["model"]["dynamics"]["enableFlexibleModel"] = True

        ### Set the stepper update period and max number of iterations
        engine_options["stepper"]["iterMax"] = \
            int(self.simu_duration_max / 1.0e-4)     # Fix maximum number of iterations by second in average
        engine_options["stepper"]["timeout"] = 1.0   # (s) Max computation time of a single step
        engine_options["stepper"]["dtMax"] = 1.0e-3  # (s) Max integration timestep

        ### Add some stochasticity to the environment

        # Change ground friction and sprint-dumper contact dynamics
        engine_options['contacts']['stiffness'] = 10 ** \
            ((MAX_GROUND_STIFFNESS_LOG - MIN_GROUND_STIFFNESS_LOG) / 2 *
            self.std_ratio.get('ground', 0.0) * \
            self.rg.uniform(low=-1.0, high=1.0) +
            (MAX_GROUND_STIFFNESS_LOG + MIN_GROUND_STIFFNESS_LOG) / 2)
        engine_options['contacts']['damping'] = \
            2.0 * np.sqrt(engine_options['contacts']['stiffness'])
        engine_options['contacts']['frictionDry'] = \
            ((MAX_GROUND_FRICTION - MIN_GROUND_FRICTION) * \
            self.std_ratio.get('ground', 0.0) * self.rg.uniform() + \
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
                self._forces_impulse.append({
                    'frame_name': frame_name,
                    't': t, 'dt': 10e-3, 'F': F
                })
                self.engine.register_force_impulse(**self._forces_impulse[-1])

            # Schedule a single force profile applied on PelvisLink.
            # Internally, it relies on a linear interpolation instead
            # of a spline for the sake of efficiency, since accuracy
            # is not a big deal, and the derivative is not needed.
            n_timesteps = 50
            t_profile = np.linspace(0.0, 1.0, n_timesteps + 1)
            F_xy_profile = PeriodicGaussianProcess(
                loc=np.zeros((2, n_timesteps + 1)),
                scale=self.std_ratio['disturbance'] * \
                    F_XY_PROFILE_SCALE * np.ones(2),
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
            self._forces_profile.append({
                'frame_name': 'PelvisLink',
                'force_function': self._force_external_profile
            })
            self.engine.register_force_profile(**self._forces_profile[-1])

        ### Set the options, finally
        self.robot.set_options(robot_options)
        self.engine.set_options(engine_options)

    def _force_external_profile(self, t, q, v, F):
        """
        @brief User-specified pre- or post- processing of the external force
               profile.

        @details Typical usecases are time rescaling (1.0 second by default),
                 or changing the orientation of the force (x/y in world frame
                 by default). It could also be used for clamping the force.
        """
        t_scaled = t / (2 * self.gait_features["step_length"])
        F[:2] = self.F_xy_profile_spline(t_scaled)

    def _is_done(self):
        """
        @brief    TODO
        """
        # Simulation termination conditions:
        # - Fall detection (if the robot has a freeflyer):
        #     The freeflyer goes lower than 75% of its height in standing pose.
        # - Maximum simulation duration exceeded
        return (self.robot.has_freeflyer and
                    self.engine_py.state[2] < self._height_neutral * 0.75) or \
               (self.engine_py.t >= self.simu_duration_max)

    def _compute_reward(self):
        # @copydoc BaseJiminyRobot::_compute_reward
        reward_dict = {}

        # Define some proxies
        reward_mixture_keys = self.reward_mixture.keys()

        if 'energy' in reward_mixture_keys:
            v_mot = self.engine_py.sensors_data[encoder.type][1]
            power_consumption = sum(np.maximum(self.action_prev * v_mot, 0.0))
            power_consumption_rel = \
                power_consumption / self._power_consumption_max
            reward_dict['energy'] = - power_consumption_rel

        if 'done' in reward_mixture_keys:
            reward_dict['done'] = 1

        # Rescale the reward so that the maximum cumulative reward is
        # independent from both the engine timestep and the maximum
        # simulation duration.
        reward_dict = {k: v * (self.dt / self.simu_duration_max)
                       for k, v in reward_dict.items()}

        # Compute the total reward
        reward_total = sum([self.reward_mixture[name] * value
                            for name, value in reward_dict.items()])

        return reward_total, reward_dict

    def _compute_reward_terminal(self):
        """
        @brief Compute the reward at the end of the episode.
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

    def step(self, action):
        """
        @brief    TODO
        """
        # Perform a single simulation step
        try:
            obs, reward, done, info = super().step(action)
        except RuntimeError:
            obs = self.observation
            reward = 0.0
            done = True
            info = {'is_success': False}

        if done:
            # Extract the log data from the simulation
            self._log_data, _ = self.engine.get_log()

            # Write log file if simulation is over (debug mode only)
            if self.debug and self._steps_beyond_done == 0:
                self.engine.write_log(self.log_path)

        return obs, reward, done, info


class WalkerPDControlJiminyEnv(WalkerTorqueControlJiminyEnv):
    def __init__(self,
                 urdf_path: str,
                 toml_path: Optional[str] = None,
                 mesh_path: Optional[str] = None,
                 simu_duration_max: float = DEFAULT_SIMULATION_DURATION,
                 dt: float = DEFAULT_ENGINE_DT,
                 hlc_to_llc_ratio: int = DEFAULT_HLC_TO_LLC_RATIO,
                 reward_mixture: Optional[dict] = None,
                 std_ratio: Optional[dict] = None,
                 debug: bool = False):
        # Backup some user arguments
        self.hlc_to_llc_ratio = hlc_to_llc_ratio

        # Low-level controller buffers
        self._q_target = None
        self._v_target = None

        # Initialize the environment
        super().__init__(urdf_path, toml_path, mesh_path, simu_duration_max,
            dt, reward_mixture, std_ratio, debug)

    def _compute_command(self):
        """
        @brief    TODO
        """
        # Extract estimated motor state based on sensors data
        q_enc, v_enc = self.engine_py.sensors_data[encoder.type]

        # Compute PD command
        u = - PID_KP * ((q_enc - self._q_target) + \
            PID_KD * (v_enc - self._v_target))

        return u

    def step(self, action):
        """
        @brief    TODO
        """
        # Update target motor state
        self._q_target, self._v_target = np.split(action, 2, axis=-1)

        # Run the whole simulation in one go
        reward = 0.0
        reward_info = {k: [] for k in self.reward_mixture.keys()}
        for _ in range(self.hlc_to_llc_ratio):
            action = self._compute_command()
            obs, reward_step, done, info_step = super().step(action)
            for k, v in info_step['reward'].items():
                reward_info[k].append(v)
            reward += reward_step
            if done:
                break

        # Extract additional information
        info = {'reward': reward_info,
                't_end': self.engine_py.t,
                'is_success': self.engine_py.t >= self.simu_duration_max}

        return obs, reward, done, info
