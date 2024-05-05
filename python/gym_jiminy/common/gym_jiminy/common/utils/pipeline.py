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
from functools import partial
from typing import (
    Dict, Any, Optional, Union, Type, Sequence, Callable, TypedDict)

import toml
import gymnasium as gym

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
    """ TODO: Write documentation.
    """

    cls: Union[Type[AbstractReward], str]
    """Reward class type.

    .. note::
        Both class type or fully qualified dotted path are supported.
    """

    kwargs: Dict[str, Any]
    """Environment constructor default arguments.

    This attribute can be omitted.
    """


class EnvConfig(TypedDict, total=False):
    """ TODO: Write documentation.
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


class BlockConfig(TypedDict, total=False):
    """ TODO: Write documentation.
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
    """ TODO: Write documentation.
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
    """ TODO: Write documentation.
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
                   layers_config: Sequence[LayerConfig]
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
    def sanitize_reward_config(reward_config: RewardConfig):
        """Sanitize reward configuration in-place.

        :param reward_config: Configuration of the reward, as a dict of type
                              `RewardConfig`.
        """
        # Get reward class type
        cls = reward_config["cls"]
        if isinstance(cls, str):
            obj = locate(cls)
            assert (isinstance(obj, type) and
                    issubclass(obj, AbstractReward))
            reward_config["cls"] = obj

        # Get reward constructor keyword-arguments
        kwargs = reward_config.get("kwargs", {})

        # Special handling for `BaseMixtureReward`
        if issubclass(cls, BaseMixtureReward):
            for reward_config in kwargs["components"]:
                sanitize_reward_config(reward_config)

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
        assert (isinstance(obj, type) and
                issubclass(obj, AbstractReward))

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
                          reward_config: RewardConfig,
                          **env_kwargs: Any) -> BasePipelineWrapper:
        """Helper adding reward on top of a base environment or a pipeline
        using `ComposedJiminyEnv` wrapper.

        :param env_creator: Callable that takes optional keyword arguments as
                            input and returns an pipeline or base environment.
        :param reward_config: Configuration of the reward, as a dict of type
                              `RewardConfig`.
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
        reward = build_reward(env, reward_config)

        # Instantiate the wrapper
        return ComposedJiminyEnv(env, reward)

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

    # Compose base environment with an extra user-specified reward if any
    reward_config = env_config.get("reward")
    if reward_config is not None:
        pipeline_creator = partial(build_composition,
                                   pipeline_creator,
                                   sanitize_reward_config(reward_config))

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


def load_pipeline(fullpath: str) -> Callable[..., InterfaceJiminyEnv]:
    """Load pipeline from JSON or TOML configuration file.

    :param: Fullpath of the configuration file.
    """
    file_ext = pathlib.Path(fullpath).suffix
    with open(fullpath, 'r') as f:
        if file_ext == '.json':
            return build_pipeline(**json.load(f))
        if file_ext == '.toml':
            return build_pipeline(**toml.load(f))
    raise ValueError("Only json and toml formats are supported.")
