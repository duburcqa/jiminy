#!/usr/bin/env python

# This tool has been copied from https://github.com/vinayak-mehta/pdftopng/blob/main/scripts/wheel_repair.py
# and extended to supported hierarchical folder architecture with multiple .pyd
# to update, and to move all the DLL in a common folder *package*.lib installed
# jointly with the package itself, similarly to auditwheel on Linux platform.
#(see also https://discuss.python.org/t/delocate-auditwheel-but-for-windows/2589/9).

import os
import shutil
import pathlib
import hashlib
import zipfile
import argparse
import tempfile

import pefile
from machomachomangler.pe import redll


def hash_filename(filepath, blocksize=65536):
    # Split original filename from extension
    root, ext = os.path.splitext(filepath)
    filename = os.path.basename(root)

    # Do NOT hash filename to make it unique in the particular case of boost
    # python modules, since otherwise it will be impossible to share a common
    # registery, which is necessary for cross module interoperability.
    if "boost_python" in filepath:
        return f"{filename}{ext}"

    # Compute unique hash based on file's content
    hasher = hashlib.sha256()
    with open(filepath, "rb") as afile:
        buf = afile.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(blocksize)

    return f"{filename}-{hasher.hexdigest()[:8]}{ext}"


def find_dll_dependencies(dll_filepath, lib_dir):
    dll_deps = {}
    for entry in pefile.PE(dll_filepath).DIRECTORY_ENTRY_IMPORT:
        entry_name = entry.dll.decode("utf-8")
        if entry_name in os.listdir(lib_dir):
            dll_deps.setdefault(
                os.path.basename(dll_filepath), set()).add(entry_name)
            nested_dll_deps = find_dll_dependencies(
                os.path.join(lib_dir, entry_name), lib_dir)
            dll_deps.update(nested_dll_deps)
    return dll_deps


def mangle_filename(old_filename, new_filename, mapping):
    with open(old_filename, "rb") as f:
        buf = f.read()
    new_buf = redll(buf, mapping)
    with open(new_filename, "wb") as f:
        f.write(new_buf)


parser = argparse.ArgumentParser(
    description="Vendor in external shared library dependencies of a wheel."
)

parser.add_argument("WHEEL_FILE", type=str, help="Path to wheel file")
parser.add_argument(
    "-d", "--dll-dir", dest="DLL_DIR", type=str, help="Directory to find the DLLs"
)
parser.add_argument(
    "-w",
    "--wheel-dir",
    dest="WHEEL_DIR",
    type=str,
    help=('Directory to store delocated wheels (default: "wheelhouse/")'),
    default="wheelhouse/",
)

args = parser.parse_args()

wheel_name = os.path.basename(args.WHEEL_FILE)
repaired_wheel = os.path.join(os.path.abspath(args.WHEEL_DIR), wheel_name)

old_wheel_dir = tempfile.mkdtemp()
new_wheel_dir = tempfile.mkdtemp()
package_name = wheel_name.split("-")[0]
bundle_name = package_name + ".libs"
bundle_path = os.path.join(new_wheel_dir, bundle_name)
os.makedirs(bundle_path)

with zipfile.ZipFile(args.WHEEL_FILE, "r") as wheel:
    wheel.extractall(old_wheel_dir)
    wheel.extractall(new_wheel_dir)
    pyd_rel_paths = [os.path.normpath(path)
                     for path in wheel.namelist() if path.endswith(".pyd")]

dll_dependencies = {}
for rel_path in pyd_rel_paths:
    abs_path = os.path.join(old_wheel_dir, rel_path)
    dll_dependencies.update(find_dll_dependencies(abs_path, args.DLL_DIR))

for dll, dependencies in dll_dependencies.items():
    mapping = {}

    if dll.endswith(".pyd"):
        rel_path = next(path for path in pyd_rel_paths if path.endswith(dll))

    for dep in dependencies:
        src_path = os.path.join(args.DLL_DIR, dep)
        hashed_name = hash_filename(src_path)  # already basename
        new_path = os.path.join(bundle_path, hashed_name)
        if dll.endswith(".pyd"):
            bundle_rel_path = os.path.join(
                "..\\" * rel_path.count(os.path.sep), bundle_name)
            mapping[dep.encode("ascii")] = os.path.join(
                bundle_rel_path, hashed_name).encode("ascii")
        else:
            mapping[dep.encode("ascii")] = hashed_name.encode("ascii")
        if not os.path.exists(new_path):
            shutil.copy2(src_path, new_path)

    if dll.endswith(".pyd"):
        old_name = os.path.join(old_wheel_dir, rel_path)
        new_name = os.path.join(new_wheel_dir, rel_path)
    else:
        old_name = os.path.join(args.DLL_DIR, dll)
        hashed_name = hash_filename(old_name)  # already basename
        new_name = os.path.join(bundle_path, hashed_name)

    mangle_filename(old_name, new_name, mapping)

pathlib.Path(os.path.dirname(repaired_wheel)).mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(repaired_wheel, "w", zipfile.ZIP_DEFLATED) as new_wheel:
    for root, dirs, files in os.walk(new_wheel_dir):
        new_root = os.path.relpath(root, new_wheel_dir)
        for file in files:
            new_wheel.write(
                os.path.join(root, file), os.path.join(new_root, file))
