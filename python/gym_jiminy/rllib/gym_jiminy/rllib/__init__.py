# pylint: disable=missing-module-docstring

from typing import Dict, Any

import ray.tune.logger

from . import ppo
from . import utilities
from . import callbacks
from . import curriculum


# Patch flatten dict method that is deep-copying the input dict in the original
# implementation instead of performing a shallow copy, which is slowing down
# down the optimization very significantly for no reason since tensorboard is
# already copying the data internally.
def _flatten_dict(dt: Dict[str, Any],
                  delimiter: str = "/",
                  prevent_delimiter: bool = False) -> Dict[str, Any]:
    """Must be patched to use copy instead of deepcopy to prevent memory
    allocation, significantly impeding computational efficiency of `TBXLogger`,
    and slowing down the optimization by about 25%.
    """
    dt = dt.copy()
    if prevent_delimiter and any(delimiter in key for key in dt):
        # Raise if delimiter is any of the keys
        raise ValueError(
            "Found delimiter `{}` in key when trying to flatten array."
            "Please avoid using the delimiter in your specification.")
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    if prevent_delimiter and delimiter in subkey:
                        # Raise  if delimiter is in any of the subkeys
                        raise ValueError(
                            "Found delimiter `{}` in key when trying to "
                            "flatten array. Please avoid using the delimiter "
                            "in your specification.")
                    add[delimiter.join([key, str(subkey)])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt

ray.tune.logger.flatten_dict = _flatten_dict  # noqa


__all__ = [
    "ppo",
    "utilities",
    "callbacks",
    "curriculum"
]
