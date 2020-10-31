import numpy as np
from collections import OrderedDict
from typing import Optional, Union, Tuple, Dict, Any, Type

import gym
from gym.spaces.utils import flatten, flatten_space

import jiminy_py.core as jiminy
from jiminy_py.core import EncoderSensor as encoder
from jiminy_py.simulator import Simulator

from .utils import (_is_breakpoint, _clamp, set_zeros, register_variables,
                    SpaceDictRecursive, FieldDictRecursive)
from .generic_bases import ControlInterface
from .env_bases import BaseJiminyEnv


class BlockInterface:
    r"""Base class for blocks used for pipeline control design.

    Block can be either observers and controllers. A block can be connected to
    any number of subsequent blocks, or directly to a `BaseJiminyEnv`
    environment.
    """
    def __init__(self):
        """Initialize the block interface.

        It only allocates some attributes.
        """
        # Define some attributes
        self.env = None
        self.observation_space = None
        self.action_space = None

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__()

    @property
    def robot(self) -> Optional[jiminy.Robot]:
        if self.env is not None:
            return self.env.robot
        else:
            return None

    @property
    def system_state(self) -> Optional[jiminy.SystemState]:
        if self.env is not None:
            return self.env.simulator.engine.system_state
        else:
            return None

    def reset(self, env: BaseJiminyEnv) -> None:
        """Reset the block for a given environment.

        .. note::
            The environment itself is not necessarily directly connected to
            this block since it may actually be connected to another block
            instead.

        .. warning::
            This method that must not be overloaded. `_setup` is the
            unique entry-point to customize the block's initialization.

        :param env: Environment.
        """
        # Backup the environment
        self.env = env

        # Configure the block
        self._setup()

        # Refresh the observation and action spaces
        self._refresh_observation_space()
        self._refresh_action_space()

    # methods to override:
    # ----------------------------

    def _setup(self) -> None:
        """Configure the block.

        .. note::
            Note that the environment `env` has already been fully initialized
            at this point, so that each of its internal buffers is up-to-date,
            but the simulation is not running yet. As a result, it is still
            possible to update the configuration of the simulator, and for
            example, to register some extra variables to monitor the internal
            state of the block.
        """
        pass

    def _refresh_observation_space(self) -> None:
        """Configure the observation of the block.

        .. note::
            This method is called right after `_setup`, so that both the
            environment and this block should be already initialized.
        """
        return NotImplementedError

    def _refresh_action_space(self) -> None:
        """Configure the action of the block.

        .. note::
            This method is called right after `_setup`, so that both the
            environment and this block should be already initialized.
        """
        return NotImplementedError


class BaseControllerBlock(BlockInterface, ControlInterface):
    r"""Base class to implement controller that can be used compute targets to
    apply to the robot of a `BaseJiminyEnv` environment, through any number of
    lower-level controllers.

    .. aafig::
        :proportional:
        :textual:

                   +----------+
        "act_ctrl" |          |
         --------->+  "ctrl"  +--------->
                   |          | "cmd_ctrl / act_env"
                   +----------+

    Formally, a controller is defined as a block mapping any action space
    'act_ctrl' to the action space of the lower-level controller 'cmd_ctrl',
    if any, and ultimately to the one of the associated environment 'act_env',
    ie the motors efforts to apply on the robot.

    The update period of the controller must be higher than the control update
    period of the environment, but both can be infinite, ie time-continuous.
    """
    def __init__(self, update_ratio: int = 1, **kwargs):
        """
        .. note::
            The space in which the command must be contained is completely
            determined by the action space of the next block (another
            controller or the environment to ultimately control). Thus, it does
            not have to be defined explicitely.

            On the contrary, the action space of the controller 'action_ctrl'
            is free and it is up to the user to define it.

        :param update_ratio: Ratio between the update period of the high-level
                             controller and the one of the subsequent
                             lower-level controller. More precisely, at each
                             simulation step, controller's action is updated
                             only once, while the lower-level command is
                             updated 'update_ratio' times.
        :param kwargs: Extra keyword arguments that may be useful for dervied
                       controller with multiple inheritance, and to allow
                       automatic pipeline wrapper generation.
        """
        # Initialize the block and control interface
        super().__init__()

        # Backup some user arguments
        self.update_ratio = update_ratio

    def _refresh_observation_space(self) -> None:
        """Configure the observation space of the controller.

        It does nothing but to return the observation space of the environment
        since it is only affecting the action space.

        .. warning::
            This method that must not be overloaded. If one need to overload
            it, when using `BaseObserverBlock` or `BlockInterface` directly
            is probably the way to go.
        """
        self.observation_space = self.env.observation_space

    # methods to override:
    # ----------------------------

    def get_fieldnames(self) -> FieldDictRecursive:
        """Get mapping between each scalar element of the action space of the
        controller and the associated fieldname for logging.

        It is expected to return an object with the same structure than the
        action space, the difference being numerical arrays replaced by lists
        of string.

        .. note::
            This method is not supposed to be called before `reset`, so that
            the controller should be already initialized at this point.
        """


BaseControllerBlock._setup.__doc__ = \
    """Configure the controller.

    It includes:

        - refreshing the action space of the controller
        - allocating memory of the controller's internal state and
          initializing it

    .. note::
        Note that the environment to ultimately control `env` has already
        been fully initialized at this point, so that each of its internal
        buffers is up-to-date, but the simulation is not running yet. As a
        result, it is still possible to update the configuration of the
        simulator, and for example, to register some extra variables to
        monitor the internal state of the controller.
    """

BaseControllerBlock._refresh_action_space.__doc__ = \
    """Configure the action space of the controller.

    .. note::
        This method is called right after `_setup`, so that both the
        environment to control and the controller itself should be already
        initialized.
    """

BaseControllerBlock.compute_command.__doc__ = \
    """Compute the action to perform by the subsequent block, namely a
    lower-level controller, if any, or the environment to ultimately
    control, based on a given high-level action.

    .. note::
        The controller is supposed to be already fully configured whenever
        this method might be called. Thus it can only be called manually
        after `reset`.  This method has to deal with the initialization of
        the internal state, but `_setup` method does so.

    :param action: Action to perform.
    """


class PDController(BaseControllerBlock):
    """Low-level Proportional-Derivative controller.

    .. warning::
        It must be connected directly to the environment to control without
        intermediary controllers.
    """
    def __init__(self,
                 update_ratio: int = 1,
                 pid_kp: Union[float, np.ndarray] = 0.0,
                 pid_kd: Union[float, np.ndarray] = 0.0,
                 **kwargs):
        """
        :param update_ratio: Ratio between the update period of the controller
                             and the one of the subsequent controller.
        :param pid_kp: PD controller position-proportional gain in motor order.
        :param pid_kd: PD controller velocity-proportional gain in motor order.
        :param kwargs: Used arguments to allow automatic pipeline wrapper
                       generation.
        """
        # Initialize the controller
        super().__init__(update_ratio)

        # Backup some user arguments
        self.pid_kp = pid_kp
        self.pid_kd = pid_kd

        # Low-level controller buffers
        self.motor_to_encoder = None
        self._q_target = None
        self._v_target = None

    def _refresh_action_space(self) -> None:
        """Configure the action space of the controller.

        The action spaces corresponds to the position and velocity of motors
        instead of the torque.
        """
        # Extract the position and velocity bounds for the observation space
        encoder_space = self.env._get_sensors_space()[encoder.type]
        pos_high, vel_high = encoder_space.high
        pos_low, vel_low = encoder_space.low

        # Reorder the position and velocity bounds to match motors order
        pos_high = pos_high[self.motor_to_encoder]
        pos_low = pos_low[self.motor_to_encoder]
        vel_high = vel_high[self.motor_to_encoder]
        vel_low = vel_low[self.motor_to_encoder]

        # Set the action space. Note that it is flattened.
        self.action_space = gym.spaces.Dict([
            (encoder.fieldnames[0], gym.spaces.Box(
                low=pos_low, high=pos_high, dtype=np.float32)),
            (encoder.fieldnames[1], gym.spaces.Box(
                low=vel_low, high=vel_high, dtype=np.float32))])

    def _setup(self) -> None:
        """Configure the controller.

        It updates the mapping from motors to encoders indices.
        """
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

        # Initialize the internal state
        motors_position_idx = sum(self.robot.motors_position_idx, [])
        self._q_target = self.system_state.q[motors_position_idx]
        self._v_target = self.system_state.v[self.robot.motors_velocity_idx]

    def get_fieldnames(self) -> FieldDictRecursive:
        pos_fieldnames = ["targetPosition" + name
                          for name in self.robot.motors_names]
        vel_fieldnames = ["targetVelocity" + name
                          for name in self.robot.motors_names]
        return OrderedDict([(encoder.fieldnames[0], pos_fieldnames),
                            (encoder.fieldnames[1], vel_fieldnames)])

    def compute_command(self,
                        action: SpaceDictRecursive
                        ) -> np.ndarray:
        """Compute the motor torques using a PD controller.

        It is proportional to the error between the measured motors positions/
        velocities and the target ones.

        :param action: Target motors positions and velocities as a vector.
                       `None` to return zero motor torques vector.
        """
        # Update the internal state of the controller
        self._q_target = action[encoder.fieldnames[0]]
        self._v_target = action[encoder.fieldnames[1]]

        # Estimate position and motor velocity from encoder data
        encoders_data = self.robot.sensors_data[encoder.type]
        q_measured, v_measured = encoders_data[:, self.motor_to_encoder]

        # Compute the joint tracking error
        q_error = q_measured - self._q_target
        v_error = v_measured - self._v_target

        # Compute PD command
        return - self.pid_kp * (q_error + self.pid_kd * v_error)


class ControlledJiminyEnv(gym.Wrapper):
    """Wrap a `BaseJiminyEnv` Gym environment and a single controller, so that
    it appears as a single, unified, environment. Eventually, the environment
    can already be wrapped inside one or several `gym.Wrapper` containers.

    If several successive controllers must be used, just wrap successively each
    controller one by one with the resulting `ControlledJiminyEnv`.

    .. aafig::
        :proportional:
        :textual:

                +---------+                    +----------+                      +----------+
        "act_1" |         |            "act_2" |          |              "act_3" |          |
        ------->+  "env"  +------> "+" ------->+ "ctrl_1" +--------> "+" ------->+ "ctrl_2" +--------> "="
                |         | "obs"              |          | "cmd_1"              |          | "cmd_2"
                +---------+                    +----------+                      +----------+

                      +-----------------------------------+
                      |                                   |
                      v                                   |
                +-----+----+         +----------+         |   +---------+
        "act_3" |          | "act_2" |          | "act_1" |   |         |
        ------->+ "ctrl_2" +-------->+ "ctrl_1" +---------o-->+  "env"  +--o------------------------->
                |          | "cmd_2" |          | "cmd_1"     |         |  | "obs or {obs + [cmd_1...] + [cmd_2...]}"
                +----------+         +-----+----+             +---------+  |
                                           ^                               |
                                           |                               |
                                           +-------------------------------+

    The output command 'cmd_X' of 'ctrl_X' must be consistent with the action
    space 'act_X' of the subsequent block. The action space of the outcoming
    unified environment will be the action space of the highest-level
    controller 'act_N', while its observation space will be the one of the
    unwrapped environment 'obs'. Alternatively, the later can also gather the
    (stacked) action space of the successive controllers if one is to observe
    the intermediary controllers' targets.

    .. note::
        The environment and each controller has its own step period. The global
        step period of the pipe will be their GCD (Greatest Common Divisor).

    .. warning::
        This design is not suitable for learning the controllers 'ctrl_X', but
        rather for robotic-oriented controllers, such as PID control, inverse
        kinematics, admittance control, or Model Predictive Control (MPC). It
        is recommended to add the controllers into the policy itself if it has
        to be trainable.
    """  # noqa: E501
    def __init__(self,
                 env: Union[gym.Wrapper, BaseJiminyEnv],
                 controller: BaseControllerBlock,
                 observe_target: bool = False):
        """
        .. note::
            As a reminder, `env.dt` refers to the learning step period,
            namely the timestep between two successive samples:

                [obs, reward, done, info]

            This definition remains true, independently of whether or not the
            environment is wrapped with a controller using this class. On the
            contrary, `env.controller_dt` corresponds to the apparent control
            update period, namely the update period of the higher-level
            controller if multiple are piped together.

        :param env: Environment to control. It can be an already controlled
                    environment wrapped in `ControlledJiminyEnv` if one desires
                    to stack several controllers with `BaseJiminyEnv`.
        :param controller: Controller to use to send targets to the subsequent
                           block.
        :param observe_target: Whether or not to gather the target of the
                               controller with the observation of the
                               environment. This option is only available if
                               the observation space is of type
                               `gym.spaces.Dict` or `gym.spaces.Box`.
                               Optional: disable by default.
        """
        # Make sure the environment actually derive for BaseJiminyEnv
        assert isinstance(env.unwrapped, BaseJiminyEnv), (
            "env.unwrapped must derived from `BaseJiminyEnv`.")

        # Backup user arguments
        self.controller = controller
        self.observe_target = observe_target
        self.controller_dt = env.controller_dt * controller.update_ratio
        self.debug = None
        self._observation = None
        self._action = None
        self._target = None
        self._command = None
        self._dt_eps = None
        self._ctrl_name = None

        # Initialize base wrapper
        super().__init__(env)

        # Reset the unified environment
        self.reset()

    @property
    def simulator(self) -> Optional[Simulator]:
        if self.env is not None:
            return self.env.unwrapped.simulator
        else:
            return None

    def _send_command(self,
                      t: float,
                      q: np.ndarray,
                      v: np.ndarray,
                      sensors_data: jiminy.sensorsData,
                      u_command: np.ndarray) -> None:
        """This method implement the callback function required by Jiminy
        Controller to get the command. In practice, it only updates a variable
        shared between C++ and Python to the internal value stored by this
        class.

        .. warning::
            This is a hidden function that is not listed as part of the member
            methods of the class. It is not intended to be called manually.

        :meta private:
        """
        u_command[:] = self.compute_command(self._action)

    def compute_command(self,
                        action: SpaceDictRecursive
                        ) -> SpaceDictRecursive:
        """Compute the motors efforts to apply on the robot.

        In practice, it updates, whenever it is necessary:

            - the target sent to the subsequent block by the controller
            - the command send to the robot by the environment through the
              subsequent block

        .. warning::
            This method is not meant to be overloaded.

        :param action: Next high-level action to perform.
        """
        # Get the current time
        t = self.simulator.stepper_state.t

        # Update the target to send to the subsequent block if necessary
        if _is_breakpoint(t, self.controller_dt, self._dt_eps):
            self._target[:] = self.controller.compute_command(action)

        # Update the command to send to the actuators of the robot if necessary
        if _is_breakpoint(t, self.env.controller_dt, self._dt_eps):
            self._command[:] = self.env.compute_command(self._target)

        return self._command

    def _compute_obs(self,
                     env_obs: SpaceDictRecursive) -> SpaceDictRecursive:
        """Compute the unified observation based on the current wrapped
        environment's observation and controller's target.

        It gathers the actual observation from the environment with the target
        of the controller, if requested, otherwise it forwards the observation
        directly without any further processing.

        .. warning::
            Beware it updates and returns the input environment observation
            whenever it is possible for the sake of efficiency.

        :param env_obs: Original observation from the environment.

        :returns: Updated environment's observation with the controller's
                  target appended.
        """
        if self.observe_target:
            if isinstance(self.env.observation_space, gym.spaces.Dict):
                env_obs.setdefault('targets', OrderedDict())[
                    self._ctrl_name] = self._action
            else:
                action_flat = flatten(
                    self.controller.action_space, self._action)
                env_obs = np.concatenate((env_obs, action_flat))
        return env_obs

    def get_obs(self) -> SpaceDictRecursive:
        """Post-processed observation.

        It clamps the observation to make sure it does not violate the lower
        and upper bounds.

        .. warning::
            This method is not meant to be overloaded.
        """
        if self.observe_target:
            return _clamp(self.observation_space, self._observation)
        return self._observation

    def reset(self) -> SpaceDictRecursive:
        """Reset the unified environment.

        In practice, it resets first the wrapped environment, next comes the
        controller, the observation space, and finally the low-level simulator
        controller.
        """
        # Reset the environment
        env_obs = self.env.reset()
        self.debug = self.env.debug

        # Reset the controller
        self.controller.reset(self.env)

        # Set the controller name.
        # It corresponds to the number of subsequent control blocks.
        i = 0
        env = self.env
        while isinstance(self.env, ControlledJiminyEnv):
            i += 1
            env = env.env
        self._ctrl_name = f"ctrl_{i}"

        # Update the action space
        self.action_space = self.controller.action_space

        # Initialize the controller's input action and output target
        self._action = self.action_space.sample()
        set_zeros(self._action)
        self._target = self.controller.compute_command(self._action)

        # Check that the initial action of the controller is consistent with
        # the action space of the environment.
        assert self.env.action_space.contains(self._target), (
            "The command is not consistent with the action space of the "
            "subsequent block.")

        # Initialize the command to apply on the robot
        self._command = self.env.compute_command(self._target)

        # Backup the temporal resolution of simulator steps
        engine_options = self.simulator.engine.get_options()
        self._dt_eps = 1.0 / engine_options["telemetry"]["timeUnit"]

        # Enforce the low-level controller.
        # Note that altering the original controller of the wrapped environment
        # is possible since it is systematically re-initialized at reset. So
        # one can restore a valid state for the environment after unwrapping it
        # simply calling `reset` method.
        self.simulator.controller.set_controller_handle(self._send_command)

        # Register the controller target to the telemetry
        if self.debug:
            register_variables(
                self.simulator.controller, self.controller.get_fieldnames(),
                self._action, self._ctrl_name)

        # Check that 'observe_target' can be enabled
        assert not self.observe_target or isinstance(
            self.env.observation_space, (gym.spaces.Dict, gym.spaces.Box)), (
            "'observe_target' is only available for environments whose the."
            "observation space inherits from `gym.spaces.Dict` or "
            "`gym.spaces.Box`.")

        # Append the controller's target to the observation if requested
        self.observation_space = self.env.observation_space
        if self.observe_target:
            if isinstance(self.env.observation_space, gym.spaces.Dict):
                self.observation_space.spaces['targets'] = \
                    self.controller.action_space
            else:
                action_space_flat = flatten_space(self.controller.action_space)
                self.observation_space = gym.spaces.Box(
                    low=np.concatenate((
                        self.observation_space.low, action_space_flat.low)),
                    high=np.concatenate((
                        self.observation_space.high, action_space_flat.high)),
                    dtype=np.float32)

        # Compute the unified observation
        self._observation = self._compute_obs(env_obs)
        return self.get_obs()

    def step(self,
             action: Optional[np.ndarray] = None
             ) -> Tuple[SpaceDictRecursive, float, bool, Dict[str, Any]]:
        """Run a simulation step for a given action.

        :param action: Next action to perform. `None` to not update it.

        :returns: Next observation, reward, status of the episode (done or
                  not), and a dictionary of extra information
        """
        # Backup the action to perform, if any
        if action is not None:
            self._action = action

        # Compute the next learning step
        env_obs, reward, done, info = self.env.step()

        # Compute the unified observation
        self._observation = self._compute_obs(env_obs)

        return self.get_obs(), reward, done, info


def build_controlled_env(env_class: Type[Union[gym.Wrapper, BaseJiminyEnv]],
                         controller_class: Type[BaseControllerBlock],
                         observe_target: bool = False,
                         **kwargs_default
                         ) -> Type[ControlledJiminyEnv]:
    """Generate a class inheriting from `ControlledJiminyEnv` and wrapping a
    given type of environment and controller together.

    .. warning::
        The arguments are passed to the environment and controller constructors
        by keyword only. So make sure there is no collision between their
        keywords arguments.

    :param env_class: Type of environment to control.
    :param controller_class: Type of controller to use to send targets to the
                             subsequent block.
    :param observe_target: Whether or not to gather the target of the
                           controller with the observation of the environment.
                           This option is only available if the observation
                           space is of type `gym.spaces.Dict` or
                           `gym.spaces.Box`.
                           Optional: disable by default.
    :param kwargs_default: Keyword arguments to forward systematically to the
                           the constructor of both the wrapped environment and
                           the controller. Note that it will only overwrite the
                           default value, and it will still be possible to set
                           different values by explicitly defining them when
                           calling the constructor of the generated wrapper.
    """
    controlled_env_class = type(
        f"{controller_class.__name__}{env_class.__name__}",
        (ControlledJiminyEnv,),
        {})

    def __init__(self, **kwargs):
        """
        :param kwargs: Keyword arguments to forward to both the wrapped
                       environment and the controller. It will overwrite
                       default values defined when creating this class, if any.
        """
        kwargs = {**kwargs_default, **kwargs}
        env = env_class(**kwargs)
        controller = controller_class(**kwargs)
        super(controlled_env_class, self).__init__(
            env, controller, observe_target)

    controlled_env_class.__init__ = __init__

    return controlled_env_class
