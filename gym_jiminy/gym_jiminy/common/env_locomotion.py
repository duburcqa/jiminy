## @file

import os
import math
import tempfile
import numpy as np
import numba as nb
from pathlib import Path
from collections import OrderedDict
from functools import lru_cache
from scipy.interpolate import make_interp_spline
from typing import List, Dict, Optional, Callable

import torch
from gym import core, spaces
from gym.utils import seeding

from jiminy_py import core as jiminy
from jiminy_py.core import (EncoderSensor as encoder,
                            EffortSensor as effort,
                            ContactSensor as contact,
                            ForceSensor as force,
                            ImuSensor as imu)
from jiminy_py.viewer import Viewer, play_logfiles
from jiminy_py.dynamics import update_quantities, \
                               get_body_world_transform, \
                               compute_freeflyer_state_from_fixed_body
from jiminy_py.robot import BaseJiminyRobot

from pinocchio import Quaternion, neutral
from pinocchio.rpy import matrixToRpy, rpyToMatrix

from .env_base import BaseJiminyEnv
from .distributions import PeriodicGaussianProcess


MIN_GROUND_STIFFNESS_LOG = 5.5
MAX_GROUND_STIFFNESS_LOG = 7.5
MIN_GROUND_FRICTION = 0.8
MAX_GROUND_FRICTION = 8.0

F_XY_IMPULSE_SCALE = 1000.0
F_XY_PROFILE_SCALE = 50.0
FLEX_STIFFNESS_SCALE = 1000
FLEX_DAMPING_SCALE = 10
ENCODER_DELAY_SCALE = 3.0e-3
ENCODER_NOISE_SCALE = np.array([0.0, 0.02], dtype=np.float64)
IMU_NOISE_SCALE = np.array(
    [0.0, 0.0, 0.0, 0.02, 0.02, 0.02, 0.2, 0.2, 0.2], dtype=np.float64)
FORCE_NOISE_SCALE = np.array([0.0, 0.0, 2.0], dtype=np.float64)

DEFAULT_ENGINE_DT = 1.0e-3  # Stepper update period

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
                 simu_duration_max: float,
                 dt: float = DEFAULT_ENGINE_DT,
                 urdf_path: str = None,
                 toml_path: Optional[str] = None,
                 mesh_path: Optional[str] = None,
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
        self.log_path = None
        self.forces_impulse = None
        self.forces_profile = None
        self._total_weight = None
        self._power_consumption_max = None
        self._quat_imu_frame_inv = None
        self._F_xy_profile_spline = None
        self._log_data = None

        # Configure and initialize the learning environment
        super().__init__(None, dt, debug)

    def __del__(self):
        if not self.debug:
            if self.robot_name != "atalante":
                os.remove(self.urdf_path)

    def _setup_environment(self):
        # Make sure a valid engine is available
        if self.engine_py is None:
            # Instantiate a new engine
            self.engine_py = BaseJiminyEngine(
                self.urdf_path, self.toml_path, self.mesh_path,
                has_freeflyer = True, use_theoretical_model = False)

            # Set the log path
            if debug:
                robot_name = self.robot.pinocchio_model.name
                self.log_path = os.path.join(
                    tempfile.gettempdir(), f"log_{robot_name}.data")

        # Remove already register forces
        self._forces_impulse = []
        self._forces_profile = []
        self.engine.remove_forces()

        # Discard log data since no longer relevant
        self._log_data = None

        # Compute the weight of the robot and the IMUs frame rotation.
        # It will be used later for computing the reward.
        total_mass = self.robot.pinocchio_data_th.mass[0]
        gravity = - self.robot.pinocchio_model_th.gravity.linear[2]
        self._total_weight = total_mass * gravity

        motor_effort_limit = self.robot.effort_limit[
            self.robot.motors_velocity_idx]
        motor_velocity_limit = self.robot.velocity_limit[
            self.robot.motors_velocity_idx]
        self._power_consumption_max = sum(
            motor_effort_limit * motor_velocity_limit)

        self._quat_imu_frame_inv = {}
        for imu_sensor_name in self.robot.sensors_names[imu.type]:
            sensor = self.robot.get_sensor(imu.type, imu_sensor_name)
            frame_name = sensor.frame_name
            frame_idx = self.robot.pinocchio_model_th.getFrameId(frame_name)
            frame = self.robot.pinocchio_model_th.frames[frame_idx]
            frame_rot = frame.placement.rotation
            self._quat_imu_frame_inv[imu_sensor_name] = Quaternion(frame_rot.T)

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

        if 'sensors' in self.std_ratio.keys():
            # Add sensor noise, bias and delay
            encoders_options = robot_options["sensors"][enc.type]
            for sensor_options in encoders_options.values():
                sensor_options['delay'] = self.std_ratio['sensors'] * \
                    self.rg.uniform() * ENCODER_DELAY_SCALE
                sensor_options['noiseStd'] = self.std_ratio['sensors'] * \
                    self.rg.uniform() * ENCODER_NOISE_SCALE
            imus_options = robot_options["sensors"][imu.type]
            for sensor_options in imus_options.values():
                sensor_options['noiseStd'] = self.std_ratio['sensors'] * \
                    self.rg.uniform() * IMU_NOISE_SCALE
            forces_options = robot_options["sensors"][imu.type]
            for sensor_options in forces_options.values():
                sensor_options['noiseStd'] = self.std_ratio['sensors'] * \
                    self.rg.uniform() * FORCE_NOISE_SCALE

        if 'model' in self.std_ratio.keys():
            # Randomize the flexibility parameters
            dynamics_options = robot_options["model"]["dynamics"]
            for flexibility in dynamics_options["flexibilityConfig"]:
                flexibility['stiffness'] += self.std_ratio['model'] * \
                    FLEX_STIFFNESS_SCALE * self.rg.uniform(low=-1.0, high=1.0)
                flexibility['damping'] += self.std_ratio['model'] * \
                    FLEX_DAMPING_SCALE * self.rg.uniform(low=-1.0, high=1.0)

            # TODO: Add biases to the URDF model
            dynamics_options["inertiaBodiesBiasStd"] = 0.0
            dynamics_options["massBodiesBiasStd"] = 0.0
            dynamics_options["centerOfMassPositionBodiesBiasStd"] = 0.0
            dynamics_options["relativePositionBodiesBiasStd"] = 0.0

        if 'disturbance' in self.std_ratio.keys():
            # Apply the disturbance to the first frame being attached to the
            # first actual body.
            for frame in engine.robot.pinocchio_model.frames:
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
                loc=torch.zeros((2, n_timesteps + 1)),
                scale=self.std_ratio['disturbance'] * \
                    F_XY_PROFILE_SCALE * torch.ones(2),
                wavelength=torch.tensor([1.0, 1.0]),
                period=torch.tensor([1.0]),
                dt=torch.tensor([1 / n_timesteps]),
                reg=1.0e-6
            ).sample().squeeze().numpy().T
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
        # Simulation termination conditions:
        # - Fall detection (if the robot has a freeflyer):
        #     The freeflyer goes lower than 90% of its height in standing pose.
        # - Maximum simulation duration exceeded
        return (self.robot.has_freeflyer and
                    self.engine_py.state[2] < self._height_neutral * 0.9) or \
               (self.engine_py.t >= self.simu_duration_max)

    def _compute_reward(self):
        # @copydoc BaseJiminyRobot::_compute_reward
        reward = {}

        # Define some proxies
        sensors_data = self.engine_py.sensors_data
        reward_mixture_keys = self.reward_mixture.keys()

        if 'energy' in reward_mixture_keys:
            v_mot = self.engine_py.sensors_data[env.type][1]
            power_consumption = sum(np.maximum(self.action_prev * v_mot, 0.0))
            power_consumption_rel = \
                power_consumption / self._power_consumption_max
            reward['energy'] = - power_consumption_rel

        if 'done' in reward_mixture_keys:
            reward['done'] = 1

        # Rescale the reward so that the maximum cumulative reward is
        # independent from both the engine timestep and the maximum
        # simulation duration.
        reward = {k: v * (self.dt / self.simu_duration_max)
                  for k, v in reward.items()}

        return reward

    def _compute_reward_terminal(self):
        """
        @brief Compute the reward at the end of the episode.
        """
        reward = {}

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
            reward['direction'] = - frontal_displacement

        return reward

    def step(self, action):
        # Perform a single simulation step
        try:
            obs, reward_info, done, info = super().step(action)
        except RuntimeError:
            obs = self.observation
            done = True
            info = {'is_success': False}
            reward_info = {}

        if done:
            # Extract the log data from the simulation
            self._log_data, _ = self.engine.get_log()

            # Write log file if simulation is over (debug mode only)
            if self.debug and self._steps_beyond_done == 0:
                self.engine.write_log(self.log_path)

            # Add the final reward if the simulation is over
            reward_final = self._compute_reward_terminal()
            reward_info.update(reward_final)

        # Compute the total reward
        reward = sum([self.reward_mixture[name] * value
                      for name, value in reward_info.items()])

        # Extract additional information
        info.update({'reward': reward_info})

        return obs, reward, done, info


class WalkerPDControlJiminyEnv(WalkerTorqueControlJiminyEnv):
    def __init__(self,
                 simu_duration_max: float,
                 dt: float = DEFAULT_ENGINE_DT,
                 hlc_to_llc_ratio: int = 1,
                 urdf_path: str = None,
                 toml_path: Optional[str] = None,
                 mesh_path: Optional[str] = None,
                 reward_mixture: Optional[dict] = None,
                 std_ratio: Optional[dict] = None,
                 debug: bool = False):
        # Backup some user arguments
        self.hlc_to_llc_ratio = hlc_to_llc_ratio

        # Low-level controller buffers
        self._q_target = None
        self._v_target = None

        # Initialize the environment
        super().__init__(simu_duration_max, dt, hlc_to_llc_ratio, urdf_path,
            toml_path, mesh_path, reward_mixture, std_ratio, debug)

    def _compute_command(self):
        # Extract estimated motor state based on sensors data
        q_enc, v_enc = self.engine_py.sensors_data[enc.type]

        # Compute PD command
        u = - PID_KP * ((q_enc - self._q_target) + \
            PID_KD * (v_enc - self._v_target))
        return u

    def step(self, action):
        # Update target motor state
        self._q_target, self._v_target = np.split(action, 2, axis=-1)

        # Run the whole simulation in one go
        reward = 0.0
        reward_info = {k: 0.0 for k in self.reward_mixture.keys()}
        for _ in range(self.hlc_to_llc_ratio):
            action = self._compute_command()
            obs, reward_step, done, info_step = super().step(action)
            for k, v in info_step['reward'].items():
                reward_info[k] += v
            reward += reward_step
            if done:
                break

        # Extract additional information
        info = {'reward': reward_info,
                't_end': self.engine_py.t,
                'is_success': self.engine_py.t >= self.simu_duration_max}

        return obs, reward, done, info
