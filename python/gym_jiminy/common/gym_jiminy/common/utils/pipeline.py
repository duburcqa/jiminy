"""Helper methods to generate learning environment pipeline, consisting in an
bare-bone environment inheriting from `BaseJiminyEnv`, wrapped together with
any number of successive blocks as a unified environment, in Matlab Simulink
fashion.

It enables to break down a complex control architectures in many submodules,
making it easier to maintain and avoiding code duplications between use cases.
"""
import re
import json
import pathlib
from pydoc import locate
from dataclasses import asdict
from functools import partial
from typing import (
    Dict, Any, Optional, Union, Type, Sequence, Callable, TypedDict, Literal,
    cast)

import h5py
import toml
import numpy as np
import gymnasium as gym

import jiminy_py.core as jiminy
from jiminy_py.dynamics import State, Trajectory

from ..bases import (InterfaceJiminyEnv,
                     InterfaceBlock,
                     BaseControllerBlock,
                     BaseObserverBlock,
                     BasePipelineWrapper,
                     ObservedJiminyEnv,
                     ControlledJiminyEnv,
                     ComposedJiminyEnv,
                     AbstractReward,
                     BaseQuantityReward,
                     BaseMixtureReward)
from ..envs import BaseJiminyEnv


class RewardConfig(TypedDict, total=False):
    """Store information required for instantiating a given reward.

    Specifically, it is a dictionary comprising the class of the reward, which
    must derive from `AbstractReward`, and optionally some keyword-arguments
    that must be passed to its corresponding constructor.
    """

    cls: Union[Type[AbstractReward], str]
    """Reward class type.

    .. note::
        Both class type or fully qualified dotted path are supported.
    """

    kwargs: Dict[str, Any]
    """Environment constructor keyword-arguments.

    This attribute can be omitted.
    """


class TrajectoriesConfig(TypedDict, total=False):
    """Store information required for adding a database of reference
    trajectories to the environment.

    Specifically, it is a dictionary comprising a set of named trajectories as
    a dictionary whose keys are the name of the trajectories and values are
    either the trajectory itself or the path of a file storing its dump in HDF5
    format, the name of the selected trajectory, and its interpolation mode.
    """

    dataset: Dict[str, Union[str, Trajectory]]
    """Set of named trajectories as a dictionary.

    .. note::
        Both `Trajectory` objects or path (absolute or relative) are supported.
    """

    name: str
    """Name of the selected trajectory if any.

    This attribute can be omitted.
    """

    mode: Literal['raise', 'wrap', 'clip']
    """Interpolation mode of the selected trajectory if any.

    This attribute can be omitted.
    """


class EnvConfig(TypedDict, total=False):
    """Store information required for instantiating a given base environment
    and compose it with some additional reward components and termination
    conditions.

    Specifically, it is a dictionary comprising the class of the base
    environment, which must derive from `BaseJiminyEnv`, optionally some
    keyword-arguments that must be passed to its corresponding constructor, and
    eventually the configuration of some additional reward with which to
    compose the base environment.
    """

    cls: Union[Type[BaseJiminyEnv], str]
    """Environment class type.

    .. note::
        Both class type or fully qualified dotted path are supported.
    """

    kwargs: Dict[str, Any]
    """Environment constructor default arguments.

    This attribute can be omitted.
    """

    reward: RewardConfig
    """Reward configuration.

    This attribute can be omitted.
    """

    trajectories: TrajectoriesConfig
    """Reference trajectories configuration.

    This attribute can be omitted.
    """


class BlockConfig(TypedDict, total=False):
    """Store information required for instantiating a given observation or
    control block.

    Specifically, it is a dictionary comprising the class of the block, which
    must derive from `BaseControllerBlock` or `BaseObserverBlock`, and
    optionally some keyword-arguments that must be passed to its corresponding
    constructor.
    """

    cls: Union[Type[BaseControllerBlock], Type[BaseObserverBlock], str]
    """Block class type. If must derive from `BaseControllerBlock` for
    controller blocks or from `BaseObserverBlock` for observer blocks.

    .. note::
        Both class type or fully qualified dotted path are supported.
    """

    kwargs: Dict[str, Any]
    """Block constructor default arguments.

    This attribute can be omitted.
    """


class WrapperConfig(TypedDict, total=False):
    """Store information required for instantiating a given environment
    pipeline wrapper.

    Specifically, it is a dictionary comprising the class of the wrapper, which
    must derive from `BasePipelineWrapper`, and optionally some
    keyword-arguments that must be passed to its corresponding constructor.
    """

    cls: Union[Type[BasePipelineWrapper], str]
    """Wrapper class type.

    .. note::
        Both class type or fully qualified dotted path are supported.
    """

    kwargs: Dict[str, Any]
    """Wrapper constructor default arguments.

    This attribute can be omitted.
    """


class LayerConfig(TypedDict, total=False):
    """Store information required for instantiating a given environment
    pipeline layer, ie either a wrapper, or the combination of an observer /
    controller block with its corresponding wrapper.

    Specifically, it is a dictionary comprising the configuration of the block
    if any, and optionally the configuration of the reward. It is generally
    sufficient to specify either one or the other. See the documentation of the
    both fields for details.
    """

    block: BlockConfig
    """Block configuration.

    This attribute can be omitted. If so, then 'wrapper_cls' must be
    specified and must not require any block. Typically, it happens when the
    wrapper is not doing any computation on its own but just transforming the
    action or observation, e.g. stacking observation frames.
    """

    wrapper: WrapperConfig
    """Wrapper configuration.

    This attribute can be omitted. If so, then 'block' must be specified and
    must this block must be associated with a unique wrapper type to allow for
    automatic type inference. It works with any observer and controller block.
    """


def build_pipeline(env_config: EnvConfig,
                   layers_config: Sequence[LayerConfig],
                   *,
                   root_path: Optional[Union[str, pathlib.Path]] = None
                   ) -> Callable[..., InterfaceJiminyEnv]:
    """Wrap together an environment inheriting from `BaseJiminyEnv` with any
    number of layers, as a unified pipeline environment class inheriting from
    `BasePipelineWrapper`. Each layer is wrapped individually and successively.

    :param env_config:
        Configuration of the environment, as a dict of type `EnvConfig`.

    :param layers_config:
        Configuration of the blocks, as a list. The list is ordered from the
        lowest level layer to the highest, each element corresponding to the
        configuration of a individual layer, as a dict of type `LayerConfig`.
    """
    # Define helper to sanitize reward configuration
    def sanitize_reward_config(reward_config: RewardConfig) -> None:
        """Sanitize reward configuration in-place.

        :param reward_config: Configuration of the reward, as a dict of type
                              `RewardConfig`.
        """
        # Get reward class type
        cls = reward_config["cls"]
        if isinstance(cls, str):
            obj = locate(cls)
            assert isinstance(obj, type) and issubclass(obj, AbstractReward)
            reward_config["cls"] = cls = obj

        # Get reward constructor keyword-arguments
        kwargs = reward_config.get("kwargs", {})

        # Special handling for `BaseMixtureReward`
        if issubclass(cls, BaseMixtureReward):
            for component_config in kwargs["components"]:
                sanitize_reward_config(component_config)

    # Define helper to build the reward
    def build_reward(env: InterfaceJiminyEnv,
                     reward_config: RewardConfig) -> AbstractReward:
        """Instantiate a reward associated with a given environment provided
        some reward configuration.

        :param env: Base environment or pipeline wrapper to wrap.
        :param reward_config: Configuration of the reward, as a dict of type
                              `RewardConfig`.
        """
        # Get reward class type
        cls = reward_config["cls"]
        assert isinstance(cls, type) and issubclass(cls, AbstractReward)

        # Get reward constructor keyword-arguments
        kwargs = reward_config.get("kwargs", {})

        # Special handling for `BaseMixtureReward`
        if issubclass(cls, BaseMixtureReward):
            kwargs["components"] = tuple(
                build_reward(env, reward_config)
                for reward_config in kwargs["components"])

        # Special handling for `BaseQuantityReward`
        if cls is BaseQuantityReward:
            quantity_config = kwargs["quantity"]
            kwargs["quantity"] = (
                quantity_config["cls"], quantity_config["kwargs"])

        return cls(env, **kwargs)

    # Define helper to build reward
    def build_composition(env_creator: Callable[..., InterfaceJiminyEnv],
                          reward_config: Optional[RewardConfig],
                          trajectories_config: Optional[TrajectoriesConfig],
                          **env_kwargs: Any) -> InterfaceJiminyEnv:
        """Helper adding reward on top of a base environment or a pipeline
        using `ComposedJiminyEnv` wrapper.

        :param env_creator: Callable that takes optional keyword arguments as
                            input and returns an pipeline or base environment.
        :param reward_config: Configuration of the reward, as a dict of type
                              `RewardConfig`.
        :param trajectories: Set of named trajectories as a dictionary. See
                             `ComposedJiminyEnv` documentation for details.
        :param env_kwargs: Keyword arguments to forward to the constructor of
                           the wrapped environment. Note that it will only
                           overwrite the default value, so it will still be
                           possible to set different values by explicitly
                           defining them when calling the constructor of the
                           generated wrapper.
        """
        # Instantiate the environment, which may be a lower-level wrapper
        env = env_creator(**env_kwargs)

        # Instantiate the reward
        reward = None
        if reward_config is not None:
            reward = build_reward(env, reward_config)

        # Get trajectory dataset
        trajectories: Dict[str, Trajectory] = {}
        if trajectories_config is not None:
            trajectories = cast(
                Dict[str, Trajectory], trajectories_config["dataset"])

        # Instantiate the composition wrapper if necessary
        if reward or trajectories:
            env = ComposedJiminyEnv(
                env, reward=reward, trajectories=trajectories)

        # Select the reference trajectory if specified
        if trajectories_config is not None:
            name = trajectories_config.get("name")
            if name is not None:
                mode = trajectories_config.get("mode", "raise")
                env.quantities.select_trajectory(name, mode)

        return env

    # Define helper to wrap a single layer
    def build_layer(env_creator: Callable[..., InterfaceJiminyEnv],
                    wrapper_cls: Type[BasePipelineWrapper],
                    wrapper_kwargs: Dict[str, Any],
                    block_cls: Optional[Type[InterfaceBlock]],
                    block_kwargs: Dict[str, Any],
                    **env_kwargs: Any
                    ) -> BasePipelineWrapper:
        """Helper wrapping a base environment or a pipeline with additional
        layer, typically an observer or a controller.

        :param env_creator: Callable that takes optional keyword arguments as
                            input and returns an pipeline or base environment.
        :param wrapper_cls: Type of wrapper to use to gather the environment
                              and the block.
        :param wrapper_kwargs: Keyword arguments to forward to the constructor
                               of the wrapper. See 'env_kwargs'.
        :param block_cls: Type of block to connect to the environment, if
                            any. `None` to disable.
                            Optional: Disabled by default
        :param block_kwargs: Keyword arguments to forward to the constructor of
                             the wrapped block. See 'env_kwargs'.
        :param env_kwargs: Keyword arguments to forward to the constructor of
                           the wrapped environment. Note that it will only
                           overwrite the default value, so it will still be
                           possible to set different values by explicitly
                           defining them when calling the constructor of the
                           generated wrapper.
        """
        # Initialize constructor arguments
        args: Any = []

        # Instantiate the environment, which may be a lower-level wrapper
        env = env_creator(**env_kwargs)
        args.append(env)

        # Instantiate the block associated with the wrapper if any
        if block_cls is not None:
            block_name = block_kwargs.pop("name", None)
            if block_name is None:
                block_index = 0
                env_wrapper: gym.Env = env
                while isinstance(env_wrapper, BasePipelineWrapper):
                    if isinstance(env_wrapper, ControlledJiminyEnv):
                        if isinstance(env_wrapper.controller, block_cls):
                            block_index += 1
                    elif isinstance(env_wrapper, ObservedJiminyEnv):
                        if isinstance(env_wrapper.observer, block_cls):
                            block_index += 1
                    env_wrapper = env_wrapper.env
                block_name = re.sub(
                    r"([a-z\d])([A-Z])", r'\1_\2', re.sub(
                        r"([A-Z]+)([A-Z][a-z])", r'\1_\2', block_cls.__name__)
                    ).lower()
                if block_index:
                    block_name += f"_{block_index}"

            block = block_cls(block_name, env, **block_kwargs)
            args.append(block)

        # Instantiate the wrapper
        return wrapper_cls(*args, **wrapper_kwargs)

    # Define callable for instantiating the base environment
    env_cls = env_config["cls"]
    if isinstance(env_cls, str):
        obj = locate(env_cls)
        assert isinstance(obj, type) and issubclass(obj, BaseJiminyEnv)
        env_cls = obj
    pipeline_creator: Callable[..., InterfaceJiminyEnv] = partial(
        env_cls, **env_config.get("kwargs", {}))

    # Parse reward configuration
    reward_config = env_config.get("reward")
    if reward_config is not None:
        sanitize_reward_config(reward_config)

    # Parse trajectory configuration
    trajectories_config = env_config.get("trajectories")
    if trajectories_config is not None:
        trajectories = trajectories_config['dataset']
        assert isinstance(trajectories, dict)
        for name, path_or_traj in trajectories.items():
            if isinstance(path_or_traj, Trajectory):
                continue
            path = pathlib.Path(path_or_traj)
            if not path.is_absolute():
                if root_path is None:
                    raise RuntimeError(
                        "The argument 'root_path' must be provided when "
                        "specifying relative trajectory paths.")
                path = pathlib.Path(root_path) / path
            trajectories[name] = load_trajectory_from_hdf5(path)

    # Compose base environment with an extra user-specified reward
    pipeline_creator = partial(build_composition,
                               pipeline_creator,
                               reward_config,
                               trajectories_config)

    # Generate pipeline recursively
    for layer_config in layers_config:
        # Extract block and wrapper config
        block_config = layer_config.get("block") or {}
        wrapper_config = layer_config.get("wrapper") or {}

        # Make sure block and wrappers are class type and parse them if string
        block_cls = block_config.get("cls")
        block_cls_: Optional[Type[InterfaceBlock]] = None
        if isinstance(block_cls, str):
            obj = locate(block_cls)
            assert (isinstance(obj, type) and
                    issubclass(obj, InterfaceBlock))
            block_cls_ = obj
        elif block_cls is not None:
            assert issubclass(block_cls, InterfaceBlock)
            block_cls_ = block_cls
        wrapper_cls = wrapper_config.get("cls")
        wrapper_cls_: Optional[Type[BasePipelineWrapper]] = None
        if isinstance(wrapper_cls, str):
            obj = locate(wrapper_cls)
            assert (isinstance(obj, type) and
                    issubclass(obj, BasePipelineWrapper))
            wrapper_cls_ = obj
        elif wrapper_cls is not None:
            assert (isinstance(wrapper_cls, type) and
                    issubclass(wrapper_cls, BasePipelineWrapper))
            wrapper_cls_ = wrapper_cls

        # Handling of default keyword arguments
        block_kwargs = block_config.get("kwargs", {})
        wrapper_kwargs = wrapper_config.get("kwargs", {})

        # Handling of default wrapper class type
        if wrapper_cls_ is None:
            if block_cls_ is not None:
                if issubclass(block_cls_, BaseControllerBlock):
                    wrapper_cls_ = ControlledJiminyEnv
                elif issubclass(block_cls_, BaseObserverBlock):
                    wrapper_cls_ = ObservedJiminyEnv
                else:
                    raise ValueError(
                        f"Block of type '{block_cls_}' does not support "
                        "automatic default wrapper type inference. Please "
                        "specify it manually.")
            else:
                raise ValueError(
                    "Either 'block.cls' or 'wrapper.cls' must be specified.")

        # Add layer on top of the existing pipeline
        pipeline_creator = partial(build_layer,
                                   pipeline_creator,
                                   wrapper_cls_,
                                   wrapper_kwargs,
                                   block_cls_,
                                   block_kwargs)

    return pipeline_creator


def load_pipeline(fullpath: Union[str, pathlib.Path]
                  ) -> Callable[..., InterfaceJiminyEnv]:
    """Load pipeline from JSON or TOML configuration file.

    :param: Fullpath of the configuration file.
    """
    fullpath = pathlib.Path(fullpath)
    root_path, file_ext = fullpath.parent, fullpath.suffix
    with open(fullpath, 'r') as f:
        if file_ext == '.json':
            return build_pipeline(**json.load(f), root_path=root_path)
        if file_ext == '.toml':
            return build_pipeline(**toml.load(f), root_path=root_path)
    raise ValueError("Only json and toml formats are supported.")


def save_trajectory_to_hdf5(trajectory: Trajectory,
                            fullpath: Union[str, pathlib.Path]) -> None:
    """Export a trajectory object to HDF5 format.

    :param trajectory: Trajectory object to save.
    :param fullpath: Fullpath of the generated HDF5 file.
    """
    # Create HDF5 file
    hdf_obj = h5py.File(fullpath, "w")

    # Dump each state attribute that are specified for all states at once
    if trajectory.states:
        state_dict = asdict(trajectory.states[0])
        state_fields = tuple(
            key for key, value in state_dict.items() if value is not None)
        for key in state_fields:
            data = np.stack([
                getattr(state, key) for state in trajectory.states], axis=0)
            hdf_obj.create_dataset(name=f"states/{key}", data=data)

    # Dump serialized robot
    robot_data = jiminy.save_to_binary(trajectory.robot)
    dataset = hdf_obj.create_dataset(name="robot", data=np.array(robot_data))

    # Dump whether to use the theoretical model of the robot
    dataset.attrs["use_theoretical_model"] = trajectory.use_theoretical_model

    # Close the HDF5 file
    hdf_obj.close()


def load_trajectory_from_hdf5(
        fullpath: Union[str, pathlib.Path]) -> Trajectory:
    """Import a trajectory object from file in HDF5 format.

    :param fullpath: Fullpath of the HDF5 file to import.

    :returns: Loaded trajectory object.
    """
    # Open HDF5 file
    hdf_obj = h5py.File(fullpath, "r")

    # Get all state attributes that are specified
    states_dict = {}
    if 'states' in hdf_obj.keys():
        for key, value in hdf_obj['states'].items():
            states_dict[key] = value[...]

    # Re-construct state sequence
    states = []
    for args in zip(*states_dict.values()):
        states.append(State(**dict(zip(states_dict.keys(), args))))

    # Build trajectory from data.
    # Null char '\0' must be added at the end to match original string length.
    dataset = hdf_obj['robot']
    robot_data = dataset[()]
    robot_data += b'\0' * (
        dataset.nbytes - len(robot_data))  # pylint: disable=no-member
    try:
        robot = jiminy.load_from_binary(robot_data)
    except MemoryError as e:
        raise MemoryError(
            "Impossible to build robot from serialized binary data. Make sure "
            "that data has been generated on a machine with the same hardware "
            "as this one.") from e

    # Load whether to use the theoretical model of the robot
    use_theoretical_model = dataset.attrs["use_theoretical_model"]

    # Close the HDF5 file
    hdf_obj.close()

    # Re-construct the whole trajectory
    return Trajectory(states, robot, use_theoretical_model)
