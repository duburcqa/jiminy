"""Generic environment to learn locomotion skills for legged robots using
Jiminy simulator as physics engine.
"""
from typing import Optional, Dict, Union, Callable, Any, Type, Sequence, Tuple

import numpy as np

from jiminy_py.core import (  # pylint: disable=no-name-in-module
    EncoderSensor as encoder,
    EffortSensor as effort,
    ContactSensor as contact,
    ForceSensor as force,
    ImuSensor as imu,
    PeriodicGaussianProcess,
    Robot)
from jiminy_py.simulator import Simulator

import pinocchio as pin

from ..utils import sample
from ..bases import InfoType
from .env_generic import BaseJiminyEnv


GROUND_FRICTION_RANGE = (0.2, 2.0)

F_IMPULSE_DT = 10.0e-3
F_IMPULSE_PERIOD = 2.0
F_IMPULSE_DELTA = 0.25
F_IMPULSE_SCALE = 1000.0
F_PROFILE_SCALE = 50.0
F_PROFILE_WAVELENGTH = 0.2
F_PROFILE_PERIOD = 1.0
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
    imu.type:  np.array([0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
}
SENSOR_BIAS_SCALE = {
    encoder.type:  np.array([0.0, 0.0]),
    effort.type: np.array([0.0]),
    contact.type: np.array([4.0, 4.0, 4.0, 20.0, 20.0, 20.0]),
    force.type: np.array([4.0, 4.0, 4.0]),
    imu.type:  np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.0, 0.0, 0.0])
}

DEFAULT_SIMULATION_DURATION = 30.0  # (s) Default simulation duration
DEFAULT_STEP_DT = 0.04              # (s) Stepper update period

DEFAULT_HLC_TO_LLC_RATIO = 1  # (NA)


ForceImpulseType = Dict[str, Union[str, float, np.ndarray]]
ForceProfileFunc = Callable[[float, np.ndarray, np.ndarray, np.ndarray], None]
ForceProfileType = Dict[str, Union[str, ForceProfileFunc]]


class WalkerJiminyEnv(BaseJiminyEnv):
    """Gym environment for learning locomotion skills for legged robots.

    Jiminy is used for both physics computations and rendering.

    The observation and action spaces are unchanged wrt `BaseJiminyEnv`.
    """
    reward_range: Tuple[float, float] = (0.0, 1.0)

    def __init__(self,
                 urdf_path: Optional[str],
                 hardware_path: Optional[str] = None,
                 mesh_path_dir: Optional[str] = None,
                 simulation_duration_max: float = DEFAULT_SIMULATION_DURATION,
                 step_dt: float = DEFAULT_STEP_DT,
                 enforce_bounded_spaces: bool = False,
                 reward_mixture: Optional[dict] = None,
                 std_ratio: Optional[dict] = None,
                 config_path: Optional[str] = None,
                 avoid_instable_collisions: bool = True,
                 debug: bool = False, *,
                 robot: Optional[Robot] = None,
                 viewer_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs: Any) -> None:
        r"""
        :param urdf_path: Path of the urdf model to be used for the simulation.
        :param hardware_path: Path of Jiminy hardware description toml file.
                              Optional: Looking for '\*_hardware.toml' file in
                              the same folder and with the same name.
        :param mesh_path_dir: Path to the folder containing the model meshes.
                              Optional: Env variable 'JIMINY_DATA_PATH' will be
                              used if available.
        :param simulation_duration_max: Maximum duration of a simulation before
                                        returning done.
        :param step_dt: Simulation timestep for learning.
        :param enforce_bounded_spaces:
            Whether to enforce finite bounds for the observation and action
            spaces. If so, '\*_MAX' are used whenever it is necessary.
        :param reward_mixture: Weighting factors of selected contributions to
                               total reward.
        :param std_ratio: Relative standard deviation of selected contributions
                          to environment stochasticity.
        :param config_path: Configuration toml file to import. It will be
                            imported AFTER loading the hardware description
                            file. It can be automatically generated from an
                            instance by calling `export_config_file` method.
                            Optional: Looking for '\*_options.toml' file in the
                            same folder and with the same name. If not found,
                            using default configuration.
        :param avoid_instable_collisions: Prevent numerical instabilities by
                                          replacing collision mesh by vertices
                                          of associated minimal volume bounding
                                          box, and replacing primitive box by
                                          its vertices.
        :param debug: Whether the debug mode must be activated. Doing it
                      enables telemetry recording.
        :param robot: Robot being simulated, already instantiated and
                      initialized. Build default robot using 'urdf_path',
                      'hardware_path' and 'mesh_path_dir' if omitted.
                      Optional: None by default.
        :param viewer_kwargs: Keyword arguments used to override the original
                              default values whenever a viewer is instantiated.
                              This is the only way to pass custom arguments to
                              the viewer when calling `render` method, unlike
                              `replay` which forwards extra keyword arguments.
                              Optional: None by default.
        :param kwargs: Keyword arguments to forward to `Simulator` and
                       `BaseJiminyEnv` constructors.
        """
        # Handling of default arguments
        if reward_mixture is None:
            reward_mixture = {
                'survival': 1.0,
                'direction': 0.0,
                'energy': 0.0,
                'failure': 0.0
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
        self.simulation_duration_max = simulation_duration_max
        self.reward_mixture = reward_mixture
        self.urdf_path = urdf_path
        self.mesh_path_dir = mesh_path_dir
        self.hardware_path = hardware_path
        self.config_path = config_path
        self.std_ratio = std_ratio
        self.avoid_instable_collisions = avoid_instable_collisions

        # Robot and engine internal buffers
        self._f_xy_profile = [
            PeriodicGaussianProcess(F_PROFILE_WAVELENGTH, F_PROFILE_PERIOD),
            PeriodicGaussianProcess(F_PROFILE_PERIOD, F_PROFILE_PERIOD)]
        self._power_consumption_max = 0.0
        self._height_neutral = 0.0

        # Configure the backend simulator
        if robot is None:
            assert isinstance(self.urdf_path, str)
            simulator = Simulator.build(
                urdf_path=self.urdf_path,
                hardware_path=self.hardware_path,
                mesh_path_dir=self.mesh_path_dir,
                config_path=self.config_path,
                avoid_instable_collisions=self.avoid_instable_collisions,
                debug=debug,
                viewer_kwargs=viewer_kwargs,
                **{**dict(
                    has_freeflyer=True,
                    use_theoretical_model=False),
                    **kwargs})
        else:
            # Instantiate a simulator and load the options
            simulator = Simulator(robot, viewer_kwargs=viewer_kwargs, **kwargs)
            simulator.import_options(config_path)

        # Initialize base class
        super().__init__(
            simulator, step_dt, enforce_bounded_spaces, debug, **kwargs)

    def _setup(self) -> None:
        """Configure the environment.

        It is doing the following steps, successively:

            - updates some proxies that will be used for computing the
              reward and termination condition,
            - enforce some options of the low-level robot and engine,
            - randomize the environment according to 'std_ratio'.

        .. note::
            This method is called internally by `reset` method at the very
            beginning. One must override it to implement new contributions to
            the environment stochasticity, or to create custom low-level robot
            if the model must be different for each learning episode.
        """
        # Call the base implementation
        super()._setup()

        if not self.robot.has_freeflyer:
            raise RuntimeError(
                "`WalkerJiminyEnv` only supports robots with freeflyer.")

        # Update some internal buffers used for computing the reward
        motor_effort_limit = self.robot.pinocchio_model.effortLimit[
            self.robot.motors_velocity_idx]
        motor_velocity_limit = self.robot.velocity_limit[
            self.robot.motors_velocity_idx]
        self._power_consumption_max = sum(
            motor_effort_limit * motor_velocity_limit)

        # Compute the height of the freeflyer in neutral configuration
        # TODO: Take into account the ground profile.
        q_init, _ = self._sample_state()
        self._height_neutral = q_init[2]

        # Get the options of robot and engine
        robot_options = self.robot.get_options()
        engine_options = self.simulator.engine.get_options()

        # Make sure to log at least the required data for terminal reward
        # computation and log replay.
        engine_options['telemetry']['enableConfiguration'] = True
        engine_options['telemetry']['enableVelocity'] = True
        engine_options['telemetry']['enableForceExternal'] = \
            'disturbance' in self.std_ratio.keys()

        # ============= Add some stochasticity to the environment =============

        # Change ground friction
        engine_options['contacts']['friction'] = sample(
            *GROUND_FRICTION_RANGE,
            scale=self.std_ratio.get('ground', 0.0),
            enable_log_scale=True,
            rg=self.np_random)

        # Add sensor noise, bias and delay
        if 'sensors' in self.std_ratio.keys():
            sensor_classes: Sequence[Union[
                Type[encoder], Type[effort], Type[contact], Type[force],
                Type[imu]]] = (encoder, effort, contact, force, imu)
            for sensor in sensor_classes:
                sensors_options = robot_options["sensors"][sensor.type]
                for sensor_options in sensors_options.values():
                    for name in ("delay", "jitter"):
                        sensor_options[name] = sample(
                            low=0.0,
                            high=(self.std_ratio['sensors'] *
                                  SENSOR_DELAY_SCALE[sensor.type]),
                            rg=self.np_random)
                    for name in (
                            ("bias", SENSOR_BIAS_SCALE),
                            ("noiseStd", SENSOR_NOISE_SCALE)):
                        sensor_options[name] = sample(
                            scale=(self.std_ratio['sensors'] *
                                   SENSOR_NOISE_SCALE[sensor.type]),
                            shape=(len(sensor.fieldnames),),
                            rg=self.np_random)

        # Randomize the flexibility parameters
        if 'model' in self.std_ratio.keys():
            if self.robot.is_flexible:
                dynamics_options = robot_options["model"]["dynamics"]
                for flexibility in dynamics_options["flexibilityConfig"]:
                    flexibility['stiffness'] += FLEX_STIFFNESS_SCALE * sample(
                        scale=self.std_ratio['model'], rg=self.np_random)
                    flexibility['damping'] += FLEX_DAMPING_SCALE * sample(
                        scale=self.std_ratio['model'], rg=self.np_random)

        # Apply the disturbance to the first actual body
        if 'disturbance' in self.std_ratio.keys():
            # Make sure the pinocchio model has at least one frame
            assert self.robot.pinocchio_model.nframes

            # Determine the actual root body of the kinematic tree
            frame = self.robot.pinocchio_model.frames[0]
            for frame in self.robot.pinocchio_model.frames:
                if frame.type == pin.FrameType.BODY and frame.parent == 1:
                    frame_name = frame.name
                    break
            else:
                raise RuntimeError(
                    "There is an issue with the robot model. Impossible to "
                    "determine the root joint.")

            # Schedule some external impulse forces applied on PelvisLink
            for t_ref in np.arange(
                    0.0, self.simulation_duration_max, F_IMPULSE_PERIOD)[1:]:
                t = t_ref + sample(scale=F_IMPULSE_DELTA, rg=self.np_random)
                f_xy = sample(dist='normal', shape=(2,), rg=self.np_random)
                f_xy /= np.linalg.norm(f_xy, ord=2)
                f_xy *= sample(
                    0.0, self.std_ratio['disturbance']*F_IMPULSE_SCALE,
                    rg=self.np_random)
                self.simulator.register_force_impulse(
                    frame_name, t, F_IMPULSE_DT, np.pad(f_xy, (0, 4)))

            # Schedule a single periodic force profile applied on PelvisLink
            for func in self._f_xy_profile:
                func.reset(self.np_random)
            self.simulator.register_force_profile(
                frame_name, self._force_external_profile)

        # Set the options, finally
        self.robot.set_options(robot_options)
        self.simulator.engine.set_options(engine_options)

    def _force_external_profile(self,
                                t: float,
                                q: np.ndarray,
                                v: np.ndarray,
                                wrench: np.ndarray) -> None:
        """User-specified processing of external force profiles.

        Typical usecases are time rescaling (1.0 second by default), or
        changing the orientation of the force (x/y in world frame by default).
        It could also be used for clamping the force.

        .. warning::
            Beware it updates 'wrench' by reference for the sake of efficiency.

        :param t: Current time.
        :param q: Current configuration vector of the robot.
        :param v: Current velocity vector of the robot.
        :param wrench: Force to apply on the robot as a vector (linear and
                       angular) [Fx, Fy, Fz, Mx, My, Mz].
        """
        # pylint: disable=unused-argument
        wrench[0] = F_PROFILE_SCALE * self._f_xy_profile[0](t)
        wrench[1] = F_PROFILE_SCALE * self._f_xy_profile[1](t)
        wrench[:2] *= self.std_ratio['disturbance']

    def has_terminated(self) -> Tuple[bool, bool]:
        """Determine whether the episode is over.

        It terminates (`terminated=True`) under the following conditions:

            - fall detection: the freeflyer goes lower than 75% of its height
              in neutral configuration.

        It is truncated under the following conditions:

            - observation out-of-bounds
            - maximum simulation duration exceeded

        :returns: terminated and truncated flags.
        """
        # Call base implementation
        terminated, truncated = super().has_terminated()

        # Check if the agent has successfully solved the task
        if self._system_state_q[2] < self._height_neutral * 0.5:
            terminated = True

        return terminated, truncated

    def compute_reward(self,
                       terminated: bool,
                       truncated: bool,
                       info: InfoType) -> float:
        """Compute reward at current episode state.

        It computes the reward associated with each individual contribution
        according to 'reward_mixture'.

        .. note::
            This method can be overwritten to implement new contributions to
            the reward, or to monitor more information.

        :returns: Aggregated reward.
        """
        reward_dict = info.setdefault('reward', {})

        # Define some proxies
        reward_mixture_keys = self.reward_mixture.keys()

        if 'survival' in reward_mixture_keys:
            reward_dict['survival'] = 1.0

        if 'energy' in reward_mixture_keys:
            v_mot = self.robot.sensors_data[encoder.type][1]
            command = self.system_state.command
            power_consumption = np.sum(np.maximum(command * v_mot, 0.0))
            power_consumption_rel = \
                power_consumption / self._power_consumption_max
            reward_dict['energy'] = - power_consumption_rel

        if terminated:
            if 'failure' in reward_mixture_keys:
                reward_dict['failure'] = - 1.0

            # Add a negative reward proportional to the average deviation on
            # Y-axis. It is equal to 0.0 if the frontal displacement is
            # perfectly symmetric wrt Y-axis over the whole trajectory.
            if 'direction' in reward_mixture_keys:
                frontal_displacement = abs(np.mean(self.log_data[
                    'HighLevelController.currentFreeflyerPositionTransY']))
                reward_dict['direction'] = - frontal_displacement

        # Compute the total reward
        reward_total = sum(self.reward_mixture[name] * value
                           for name, value in reward_dict.items())

        return reward_total
