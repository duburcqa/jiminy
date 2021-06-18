"""Helper methods to generate learning environment pipeline, consisting in an
barebone environment inheriting from `BaseJiminyEnv`, wrapped together with
any number of successive blocks as a unified environment, in Matlab Simulink
fashion.

It enables to break down a complex control architectures in many submodules,
making it easier to maintain and avoiding code duplications between usecases.
"""
import json
import pathlib
from pydoc import locate
from typing import Optional, Union, Dict, Any, Type, Sequence, List

import gym
import toml
from typing_extensions import TypedDict

from .envs import BaseJiminyEnv
from .bases import (BlockInterface,
                    BaseControllerBlock,
                    BaseObserverBlock,
                    BasePipelineWrapper,
                    ObservedJiminyEnv,
                    ControlledJiminyEnv)


class EnvConfig(TypedDict, total=False):
    """Environment class type.

    .. note::
        Both class type or fully qualified dotted path are supported.
    """
    env_class: Union[Type[BaseJiminyEnv], str]

    """Environment constructor default arguments.

    This attribute can be omitted.
    """
    env_kwargs: Dict[str, Any]


class BlockConfig(TypedDict, total=False):
    """Block class type. If specified, it must derive from
    `BaseControllerBlock` for controller blocks or `BaseObserverBlock` for
    observer blocks.

    This attribute can be omitted. If so, then 'block_kwargs' must be omitted
    and 'wrapper_class' must be specified. Indeed, not all block are associated
    with a dedicated observer or controller object. It happens when the block
    is not doing any computation on its own but just transforming the action or
    observation, e.g. stacking observation frames.

    .. note::
        Both class type or fully qualified dotted path are supported.
    """
    block_class: Union[
        Type[BaseControllerBlock], Type[BaseObserverBlock], str]

    """Block constructor default arguments.

    This attribute can be omitted.
    """
    block_kwargs: Dict[str, Any]

    """Wrapper class type.

    This attribute can be omitted. If so, then 'wrapper_kwargs' must be omitted
    and 'block_class' must be specified. The latter will be used to infer the
    default wrapper type.

    .. note::
        Both class type or fully qualified dotted path are supported.
    """
    wrapper_class: Union[Type[BasePipelineWrapper], str]

    """Wrapper constructor default arguments.

    This attribute can be omitted.
    """
    wrapper_kwargs: Dict[str, Any]


def build_pipeline(env_config: EnvConfig,
                   blocks_config: Sequence[BlockConfig] = ()
                   ) -> Type[BasePipelineWrapper]:
    """Wrap together an environment inheriting from `BaseJiminyEnv` with any
    number of blocks, as a unified pipeline environment class inheriting from
    `BasePipelineWrapper`. Each block is wrapped individually and successively.

    :param env_config:
        Configuration of the environment, as a dict of type `EnvConfig`.

    :param blocks_config:
        Configuration of the blocks, as a list. The list is ordered from the
        lowest level block to the highest, each element corresponding to the
        configuration of a individual block, as a dict of type `BlockConfig`.
    """
    # pylint: disable-all

    # Define helper to wrap a single block
    def _build_wrapper(env_class: Union[
                           Type[gym.Wrapper], Type[BaseJiminyEnv]],
                       env_kwargs: Optional[Dict[str, Any]] = None,
                       block_class: Optional[Union[
                           Type[BlockInterface], str]] = None,
                       block_kwargs: Optional[Dict[str, Any]] = None,
                       wrapper_class: Optional[Union[
                           Type[BasePipelineWrapper], str]] = None,
                       wrapper_kwargs: Optional[Dict[str, Any]] = None
                       ) -> Type[ControlledJiminyEnv]:
        """Generate a class inheriting from 'wrapper_class' wrapping a given
        type of environment, optionally gathered with a block.

        .. warning::
            Beware of the collision between the keywords arguments of the
            wrapped environment and block. It would be impossible to
            overwrite their default values independently.

        :param env_class: Type of environment to wrap.
        :param env_kwargs: Keyword arguments to forward to the constructor of
                           the wrapped environment. Note that it will only
                           overwrite the default value, so it will still be
                           possible to set different values by explicitly
                           defining them when calling the constructor of the
                           generated wrapper.
        :param block_class: Type of block to connect to the environment, if
                            any. `None` to disable.
                            Optional: Disabled by default
        :param block_kwargs: Keyword arguments to forward to the constructor of
                             the wrapped block. See 'env_kwargs'.
        :param wrapper_class: Type of wrapper to use to gather the environment
                              and the block.
        :param wrapper_kwargs: Keyword arguments to forward to the constructor
                               of the wrapper. See 'env_kwargs'.
        """
        # pylint: disable-all

        # Make sure block and wrappers are class type and parse them if string
        block_class_obj: Optional[Type[BlockInterface]] = None
        if isinstance(block_class, str):
            obj = locate(block_class)
            assert (isinstance(obj, type) and
                    issubclass(obj, BlockInterface))
            block_class_obj = obj
        elif block_class is not None:
            assert issubclass(block_class, BlockInterface)
            block_class_obj = block_class
        wrapper_class_obj: Optional[Type[BasePipelineWrapper]] = None
        if isinstance(wrapper_class, str):
            obj = locate(wrapper_class)
            assert (isinstance(obj, type) and
                    issubclass(obj, (gym.Wrapper, BasePipelineWrapper)))
            wrapper_class_obj = obj
        elif wrapper_class is not None:
            assert issubclass(
                wrapper_class, (gym.Wrapper, BasePipelineWrapper))
            wrapper_class_obj = wrapper_class

        # Handling of default wrapper class type
        if wrapper_class_obj is None:
            if block_class_obj is not None:
                if issubclass(block_class_obj, BaseControllerBlock):
                    wrapper_class_obj = ControlledJiminyEnv
                elif issubclass(block_class_obj, BaseObserverBlock):
                    wrapper_class_obj = ObservedJiminyEnv
                else:
                    raise ValueError(
                        f"Block of type '{block_class}' does not support "
                        "automatic default wrapper type inference. Please "
                        "specify it manually.")
            else:
                raise ValueError(
                    "Either 'block_class' or 'wrapper_class' must be "
                    "specified.")

        # Dynamically generate wrapping class
        wrapper_name = f"{wrapper_class_obj.__name__}Wrapper"
        if block_class_obj is not None:
            wrapper_name += f"{block_class_obj.__name__}Block"
        wrapped_env_class = type(wrapper_name, (wrapper_class_obj,), {})

        # Implementation of __init__ method must be done after declaration of
        # the class, because the required closure for calling `super()` is not
        # available when creating a class dynamically.
        def __init__(self: wrapped_env_class,  # type: ignore[valid-type]
                     **kwargs: Any) -> None:
            """
            :param kwargs: Keyword arguments to forward to both the wrapped
                           environment and the controller. It will overwrite
                           default values.
            """
            nonlocal env_class, env_kwargs, block_class_obj, block_kwargs, \
                wrapper_kwargs

            # Initialize constructor arguments
            args = []

            # Define the arguments related to the environment
            if env_kwargs is not None:
                env_kwargs_default = {**env_kwargs, **kwargs}
            else:
                env_kwargs_default = kwargs
            env = env_class(**env_kwargs_default)
            args.append(env)

            # Define the arguments related to the block, if any
            if block_class_obj is not None:
                if block_kwargs is not None:
                    block_kwargs_default = {**block_kwargs, **kwargs}
                else:
                    block_kwargs_default = kwargs
                args.append(block_class_obj(
                    env.unwrapped, **block_kwargs_default))

            # Define the arguments related to the wrapper
            if wrapper_kwargs is not None:
                wrapper_kwargs_default = {**wrapper_kwargs, **kwargs}
            else:
                wrapper_kwargs_default = kwargs

            super(wrapped_env_class, self).__init__(  # type: ignore[arg-type]
               *args, **wrapper_kwargs_default)

        wrapped_env_class.__init__ = __init__  # type: ignore[misc]

        if issubclass(wrapper_class_obj, gym.Wrapper):
            # Override __dir__ method if the wrapper inherits from
            # `gym.Wrapper`, to be consistent with the custom attribute lookup.
            def __dir__(self: wrapped_env_class  # type: ignore[valid-type]
                        ) -> List[str]:
                """Attribute lookup.

                It is mainly used by autocomplete feature of Ipython. It is
                overloaded to get consistent autocompletion wrt `getattr`.
                """
                wrapper_names = super(  # type: ignore[arg-type]
                    wrapped_env_class, self).__dir__()
                env_names = [name for name in
                             self.env.__dir__()  # type: ignore[attr-defined]
                             if not name.startswith('_')]
                return wrapper_names + env_names

            wrapped_env_class.__dir__ = __dir__  # type: ignore[assignment]

        return wrapped_env_class

    # Generate pipeline sequentially
    pipeline_class = env_config['env_class']
    if isinstance(pipeline_class, str):
        obj = locate(pipeline_class)
        assert (isinstance(obj, type) and
                issubclass(obj, (gym.Wrapper, BaseJiminyEnv)))
        pipeline_class = obj
    env_kwargs = env_config.get('env_kwargs', None)
    for config in blocks_config:
        pipeline_class = _build_wrapper(
            pipeline_class, env_kwargs, **config)
        env_kwargs = None
    return pipeline_class


def load_pipeline(fullpath: str) -> Type[BasePipelineWrapper]:
    """Load pipeline from JSON or TOML configuration file.

    :param: Fullpath of the configuration file.
    """
    file_ext = pathlib.Path(fullpath).suffix
    with open(fullpath, 'r') as f:
        if file_ext == '.json':
            return build_pipeline(**json.load(f))
        elif file_ext == '.toml':
            return build_pipeline(**toml.load(f))
    raise ValueError("Only json and toml formats are supported.")
