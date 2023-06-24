"""Helper methods to generate learning environment pipeline, consisting in an
bare-bone environment inheriting from `BaseJiminyEnv`, wrapped together with
any number of successive blocks as a unified environment, in Matlab Simulink
fashion.

It enables to break down a complex control architectures in many submodules,
making it easier to maintain and avoiding code duplications between use cases.
"""
import json
import pathlib
from pydoc import locate
from itertools import chain
from typing import (
    Dict, Any, Optional, Union, Type, Sequence, Iterable, TypedDict, TypeVar)

import toml
import gymnasium as gym

from .bases import (JiminyEnvInterface,
                    BlockInterface,
                    BaseControllerBlock,
                    BaseObserverBlock,
                    BasePipelineWrapper,
                    ObservedJiminyEnv,
                    ControlledJiminyEnv)
from .envs import BaseJiminyEnv


PipelineWrapperType = TypeVar('PipelineWrapperType', bound=BasePipelineWrapper)


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
    def _build_wrapper(env_class: Type[JiminyEnvInterface],
                       env_kwargs: Optional[Dict[str, Any]] = None,
                       block_class: Optional[Union[
                           Type[BlockInterface], str]] = None,
                       block_kwargs: Optional[Dict[str, Any]] = None,
                       wrapper_class: Optional[Union[
                           Type[BasePipelineWrapper], str]] = None,
                       wrapper_kwargs: Optional[Dict[str, Any]] = None
                       ) -> Type[BasePipelineWrapper]:
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
        # Make sure block and wrappers are class type and parse them if string
        block_class_: Optional[Type[BlockInterface]] = None
        if isinstance(block_class, str):
            obj = locate(block_class)
            assert (isinstance(obj, type) and
                    issubclass(obj, BlockInterface))
            block_class_ = obj
        elif block_class is not None:
            assert issubclass(block_class, BlockInterface)
            block_class_ = block_class
        wrapper_class_: Optional[Type[BasePipelineWrapper]] = None
        if isinstance(wrapper_class, str):
            obj = locate(wrapper_class)
            assert (isinstance(obj, type) and
                    issubclass(obj, BasePipelineWrapper))
            wrapper_class_ = obj
        elif wrapper_class is not None:
            assert (isinstance(wrapper_class, type) and
                    issubclass(wrapper_class, BasePipelineWrapper))
            wrapper_class_ = wrapper_class

        # Handling of default wrapper class type
        if wrapper_class_ is None:
            if block_class_ is not None:
                if issubclass(block_class_, BaseControllerBlock):
                    wrapper_class_ = ControlledJiminyEnv
                elif issubclass(block_class_, BaseObserverBlock):
                    wrapper_class_ = ObservedJiminyEnv
                else:
                    raise ValueError(
                        f"Block of type '{block_class}' does not support "
                        "automatic default wrapper type inference. Please "
                        "specify it manually.")
            else:
                raise ValueError(
                    "Neither 'block_class' or 'wrapper_class' is specified.")

        def _init_impl(self: PipelineWrapperType, **kwargs: Any) -> None:
            """
            :param kwargs: Keyword arguments to forward to both the base
                           environment. It will overwrite default values.
            """
            nonlocal env_class, env_kwargs, block_class_, block_kwargs, \
                wrapper_kwargs

            # Initialize constructor arguments
            args: Any = []

            # Define the arguments related to the environment
            if env_kwargs is None:
                env_kwargs = {}
            env = env_class(**{**env_kwargs, **kwargs})
            args.append(env)

            # Define the arguments related to the block, if any
            if block_class_ is not None:
                if block_kwargs is None:
                    block_kwargs = {}
                block_name = block_kwargs.pop("name", None)
                if block_name is None:
                    block_index = 0
                    block_type = block_class_.type
                    env_wrapper: gym.Env = env
                    while isinstance(env_wrapper, BasePipelineWrapper):
                        if isinstance(env_wrapper, ControlledJiminyEnv):
                            if env_wrapper.controller.type == block_type:
                                block_index += 1
                        elif isinstance(env_wrapper, ObservedJiminyEnv):
                            if env_wrapper.observer.type == block_type:
                                block_index += 1
                        env_wrapper = env_wrapper.env
                    block_name = f"{block_type}_{block_index}"
                block = block_class_(block_name, env, **block_kwargs)
                args.append(block)

            # Define the arguments related to the wrapper
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            super(self.__class__, self).__init__(*args, **wrapper_kwargs)

        def _dir_impl(self: PipelineWrapperType) -> Iterable[str]:
            """Attribute lookup.

            It is mainly used by autocomplete feature of Ipython. It is
            overloaded to get consistent autocompletion wrt `getattr`.
            """
            return chain(
                super(self.__class__, self).__dir__(),
                (name for name in dir(self.env) if not name.startswith('_')))

        # Dynamically generate wrapping class
        wrapper_name = f"{wrapper_class_.__name__}Wrapper"
        if block_class_ is not None:
            wrapper_name += f"{block_class_.__name__}Block"
        wrapper_class = type(wrapper_name, (wrapper_class_,), {
            "__init__": _init_impl, "__dir__": _dir_impl})

        return wrapper_class

    # Generate pipeline sequentially
    pipeline_class: Union[
        Type[JiminyEnvInterface], str] = env_config['env_class']
    if isinstance(pipeline_class, str):
        obj = locate(pipeline_class)
        assert isinstance(obj, type) and issubclass(obj, JiminyEnvInterface)
        pipeline_class = obj
    env_kwargs = env_config.get('env_kwargs')
    for config in blocks_config:
        pipeline_class = _build_wrapper(
            pipeline_class, env_kwargs, **config)
        env_kwargs = None

    # Assert(s) for type checker
    assert issubclass(pipeline_class, BasePipelineWrapper)

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
