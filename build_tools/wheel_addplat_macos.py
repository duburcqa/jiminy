import sys
from typing import Tuple, FrozenSet

import packaging.utils
from packaging.utils import NormalizedName, BuildTag
from packaging.version import Version
from packaging.tags import Tag

from delocate.cmd.delocate_addplat import main

parse_wheel_filename_orig = packaging.utils.parse_wheel_filename

def parse_wheel_filename(
    filename: str,
) -> Tuple[NormalizedName, Version, BuildTag, FrozenSet[Tag]]:
    name, version, build, tags = parse_wheel_filename_orig(filename)
    return name.replace("-", "_"), version, build, tags

packaging.utils.parse_wheel_filename = parse_wheel_filename


if __name__ == '__main__':
    sys.exit(main())
