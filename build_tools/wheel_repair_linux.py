import os
import sys
import stat
import shutil
import itertools
from typing import Tuple

import auditwheel.repair
from auditwheel.repair import logger
from auditwheel.elfutils import elf_read_rpaths
from auditwheel.patcher import ElfPatcher
from auditwheel.main import main


copylib_orig = auditwheel.repair.copylib

def copylib(src_path: str, dest_dir: str,
            patcher: ElfPatcher) -> Tuple[str, str]:
    # Do NOT hash filename to make it unique in the particular case of boost
    # python modules, since otherwise it will be impossible to share a common
    # registry, which is necessary for cross module interoperability.
    if "libboost_python" in src_path:
        src_name = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, src_name)
        if os.path.exists(dest_path):
            return src_name, dest_path

        logger.debug('Grafting: %s -> %s', src_path, dest_path)
        shutil.copy2(src_path, dest_path)
        rpaths = elf_read_rpaths(src_path)
        statinfo = os.stat(dest_path)
        if not statinfo.st_mode & stat.S_IWRITE:
            os.chmod(dest_path, statinfo.st_mode | stat.S_IWRITE)
        patcher.set_soname(dest_path, src_name)
        if any(itertools.chain(rpaths['rpaths'], rpaths['runpaths'])):
            patcher.set_rpath(dest_path, dest_dir)

        return src_name, dest_path
    return copylib_orig(src_path, dest_dir, patcher)

auditwheel.repair.copylib = copylib


if __name__ == '__main__':
    sys.exit(main())
