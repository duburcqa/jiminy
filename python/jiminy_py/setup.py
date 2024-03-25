from glob import glob
from importlib.metadata import version
from setuptools import setup, dist, find_namespace_packages
from setuptools.command.install import install


# Force setuptools to recognize that this is actually a binary distribution
class BinaryDistribution(dist.Distribution):
    def is_pure(self) -> bool:
        return False

    def has_ext_modules(self) -> bool:
        return True


# Forcing setuptools not to consider shared libraries as purelib
class InstallPlatlib(install):
    def finalize_options(self) -> None:
        super().finalize_options()
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


# Determine the supported range of numpy versions, assuming that the version
# currently available was used to compile all C++ extension modules bundled
# with Jiminy.
# * Numpy API as limited backward compatibility range based on some complex
#   logics not following a predefined pattern (since 1.25). See documentation:
#   https://numpy.org/devdocs/dev/depending_on_numpy.html#build-time-dependency
# * Numpy API is only minor-version forward compatible
np_ver = tuple(map(int, version('numpy').split(".", 3)[:2]))
if np_ver < (1, 25):
    np_req = f"numpy>={np_ver[0]}.{np_ver[1]}.0"
    if np_ver < (1, 20):
        np_req += ",<1.20.0"
    elif np_ver < (1, 22):
        np_req += ",!=1.21.0,!=1.21.1,!=1.21.2,!=1.21.3,!=1.21.4"
else:
    if np_ver < (2, 1):
        np_req = "numpy>=1.24"  # All version down to 1.19 are supported
    else:
        raise ImportError("'numpy>2.0' not supported at built-time for now.")
    np_req += f",<{np_ver[0]}.{np_ver[1] + 1}"


setup(
    name="jiminy-py",
    version="@PROJECT_VERSION@",
    description=(
        "Fast and light weight simulator of rigid poly-articulated systems."),
    long_description=open("@SOURCE_DIR@/README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://duburcqa.github.io/jiminy/README.html",
    project_urls={
        "Documentation": "https://duburcqa.github.io/jiminy",
        "Bug Tracker": "https://github.com/duburcqa/jiminy/issues",
        "Source": "https://github.com/duburcqa/jiminy"
    },
    download_url=("https://github.com/duburcqa/jiminy/archive/"
                  "@PROJECT_VERSION@.tar.gz"),
    author="Alexis Duburcq",
    author_email="alexis.duburcq@gmail.com",
    maintainer="Alexis Duburcq",
    license="MIT",
    python_requires=">=3.8,<3.13",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ],
    keywords="robotics physics simulator",
    distclass=BinaryDistribution,
    cmdclass={
        "install": InstallPlatlib
    },
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},
    data_files=[
        ("cmake", glob('src/jiminy_py/core/cmake/*'))
    ],
    include_package_data=True,
    entry_points={"console_scripts": [
        "jiminy_plot=jiminy_py.plot:plot_log_interactive",
        ("jiminy_meshcat_server="
         "jiminy_py.viewer.meshcat.server:start_meshcat_server_standalone"),
        "jiminy_replay=jiminy_py.viewer.replay:_play_logs_files_entrypoint"
    ]},
    install_requires=[
        # Standard library for matrix algebra.
        # - 1.20 breaks ABI
        # - >=1.21,<1.21.5 is causing segfault with boost::python.
        #   See issue: https://github.com/boostorg/python/issues/376
        # - 1.22 breaks API for compiled libs.
        # - 1.24 adds `dtype` optional argument to `np.stack`
        # - 2.0 is backward compatible up to 1.23, but not forward compatible.
        #   see: https://numpy.org/devdocs/dev/depending_on_numpy.html
        np_req,
        # Parser for Jiminy's hardware description file.
        "toml",
        # Standalone cross-platform mesh visualizer used as Viewer's backend.
        # Panda3d is NOT supported by PyPy even if built from source.
        # - 1.10.12 fixes numerous bugs
        # - 1.10.13 crashes when generating wheels on MacOS
        "panda3d>=1.10.14",
        # Photo-realistic shader for Panda3d to improve rendering of meshes.
        # - 0.11.X is not backward compatible.
        "panda3d-simplepbr==0.11.2",
        # Provide helpers to make life easier with panda3d for roboticists
        "panda3d-viewer",
        # Used internally by Viewer to record video programmatically when
        # Panda3d is used as rendering backend.
        # - >= 8.0.0 provides cross-platform precompiled binary wheels
        "av>=8.0.0",
        # Used internally by Robot to replace meshes by associated minimal
        # volume bounding box.
        "trimesh",
        # Display elegant and versatile process bar.
        "tqdm"
    ],
    extras_require={
        "plot": [
            # Standard library to generate figures.
            # - 3.7.0: introduces 'outside' keyword for legend location
            "matplotlib>=3.7.0"
        ],
        "meshcat": [
            # Web-based mesh visualizer used as Viewer's backend.
            # - 0.3.2 fixes the rendering of DAE meshes
            "meshcat>=0.3.2",
            # Used to detect running Meshcat servers and avoid orphan child
            # processes.
            "psutil",
            # Low-level backend for Ipython powering Jupyter notebooks
            "ipykernel>=5.0,<7.0",
            # Used internally by Viewer to read/write Meshcat snapshots
            "pillow",
            # Used internally by Viewer to enable recording video
            # programmatically with Meshcat as backend.
            "playwright"
        ],
        "dev": [
            # Generate Python type hints files (aka. stubs) for C extensions.
            # Natively, it only supports PyBind11, but it has been patched to
            # partially support of Boost::Python (`build_tools/stub_gen.py`).
            "pybind11_stubgen<1.0",
            # Used in unit tests for checking if viewer screen captures match
            "pillow",
            # Used in unit tests for numerical integration and interpolation
            "scipy",
            # Stub for static type checking
            "types-psutil",
            "types-Pillow",
            "types-toml",
            "types-tqdm",
            # Check PEP8 conformance of Python native code
            "flake8",
            # Python linter
            "pylint>=3.1",
            # Python static type checker
            "mypy>=1.5.0",
            # Dependency for documentation generation
            "pygments",
            # Dependency for documentation generation
            "colorama",
            # Generate Python docs and render '.rst' nicely
            "sphinx>=4.0.0",
            # 'Read the Docs' Sphinx docs style
            "sphinx-rtd-theme",
            # Render markdown in sphinx docs
            "myst-parser",
            # Render Jupyter Notebooks in sphinx docs
            # v0.8.8 introduces a bug for empty 'raw' directives
            "nbsphinx!=0.8.8",
            # Render ASCII art diagram (https://aafigure.readthedocs.io)
            "aafigure",
            # Bridge between doxygen and sphinx. Used to generate C++ API docs
            "breathe",
            # Repair wheels to embed shared libraries.
            # - 5.2.1: Speed up and binary size reduction
            "auditwheel>=5.2.1",
            # Required by auditwheel but not explicitly added as dependency...
            "patchelf >= 0.14; sys_platform == 'linux'",
            # Official package for building wheels
            "build"
        ]
    },
    zip_safe=False
)
