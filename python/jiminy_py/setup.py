from setuptools import setup, dist, find_packages
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


setup(
    name="jiminy_py",
    version="@PROJECT_VERSION@",
    description=(
        "Fast and light weight simulator of rigid poly-articulated systems."),
    long_description=open("@SOURCE_DIR@/README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://duburcqa.github.io/jiminy/README.html",
    project_urls={
        "Source": "https://github.com/duburcqa/jiminy",
        "Documentation":
            "https://duburcqa.github.io/jiminy/api/jiminy_py/index.html",
        "Tutorial": "https://duburcqa.github.io/jiminy/tutorial.html"
    },
    download_url=("https://github.com/duburcqa/jiminy/archive/"
                  "@PROJECT_VERSION@.tar.gz"),
    author="Alexis Duburcq",
    author_email="alexis.duburcq@gmail.com",
    maintainer="Alexis Duburcq",
    license="MIT",
    python_requires=">=3.6,<3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ],
    keywords="robotics physics simulator",
    distclass=BinaryDistribution,
    cmdclass={
        "install": InstallPlatlib
    },
    packages=find_packages("src"),
    package_dir={"": "src"},
    data_files=[
        ("cmake", [
            "src/jiminy_py/core/cmake/jiminyConfig.cmake",
            "src/jiminy_py/core/cmake/jiminyConfigVersion.cmake"
        ])
    ],
    include_package_data=True,
    entry_points={"console_scripts": [
        "jiminy_plot=jiminy_py.log:plot_log",
        ("jiminy_meshcat_server="
         "jiminy_py.meshcat.server:start_meshcat_server_standalone"),
        "jiminy_replay=jiminy_py.viewer.replay:_play_logs_files_entrypoint"
    ]},
    install_requires=[
        # Used internally by Viewer to read/write snapshots.
        "pillow",
        # Add support of TypedDict to any Python 3 version.
        # 3.10.0 adds 'ParamSpec' that is required for pylint>=2.11.1.
        "typing_extensions>=3.10.0",
        # Display elegant and versatile process bar.
        "tqdm",
        # Standard library for matrix algebra.
        "numpy",
        # Use to operate on nested data structure conveniently
        # - 0.1.5 introduces `tree.traverse` method that it used to operate on
        # `gym.spaces.Dict`.
        # - 0.1.6 adds support of Python 3.9 and unifies API method naming.
        "dm-tree>=0.1.6",
        # Used internally for interpolation and filtering.
        # No wheel is distributed for PyPy on pypi, and pip is only able to
        # build from source after install `libatlas-base-dev` system
        # dependency.
        # 1.2.0 fixes `fmin_slsqp` optimizer returning wrong `imode` ouput.
        # 1.8.0: `scipy.spatial.qhull._Qhull` is no longer exposed.
        "scipy>=1.2.0,<1.8.0",
        # Standard library to generate figures.
        "matplotlib",
        # Used internally to read HDF5 format log files.
        # No wheel is distributed for PyPy on pypi, but pip is able to build
        # from source without additionnal dependencies.
        "h5py",
        # Used internally by Robot to replace meshes by associated minimal
        # volume bounding box.
        # No wheel is distributed for PyPy on pypi, and pip is only able to
        # build from source after install `hdf5-dev` system depdency.
        "trimesh",
        # Parser for Jiminy's hardware description file.
        "toml",
        # Web-based mesh visualizer used as Viewer's backend.
        # 0.0.18 introduces many new features, including loading generic
        # geometries and jiminy_py viewer relies on it to render collision
        # bodies.
        # 0.3.1 updates threejs from 122 to 132, breakin compatibility with
        # the old, now deprecated, geometry class used to internally to display
        # tile floor.
        # 0.3.2 fixes the rendering of DAE meshes.
        "meshcat>=0.3.2",
        # Standalone cross-platform mesh visualizer used as Viewer's backend.
        # 1.10.9 adds support of Nvidia EGL rendering without X11 server.
        # Panda3d is NOT supported by PyPy and cannot be built from source.
        # 1.10.10 fixes an impressive list of bugs.
        "panda3d>=1.10.10",
        # Provide helper methods and class to make it easier to use panda3d for
        # robotic applications.
        "panda3d_viewer",
        # Photo-realistic shader for Panda3d to improve rendering of meshes.
        "panda3d_simplepbr",
        # Used internally by Viewer to record video programmatically when
        # Meshcat is not used as rendering backend.
        # Cross-platform precompiled binary wheels are provided since 8.0.0.
        "av",
        # Used internally by Viewer to detect running Meshcat servers and avoid
        # orphan child processes.
        "psutil",
        # Used internally by Viewer to enable recording video programmatically
        # while using Meshcat as backend.
        "pyppeteer",
        # Used internally by Viewer to send/receive Javascript requests while
        # recording video using Meshcat backend.
        # `HTMLSession` is available since 0.3.4.
        "requests_html>=0.3.4"
    ],
    extras_require={
      "dev": [
          # Stub for static type checking
          "types-toml",
          # Check PEP8 conformance of Python native code
          "flake8",
          # Python linter
          "pylint>=2.12.2",
          # Python static type checker
          "mypy>=0.931",
          # Fix dependency issue with 'sphinx'
          "jinja2>=3.0,<3.1",
          # Generate Python docs and render '.rst' nicely
          "sphinx",
          # 'Read the Docs' Sphinx docs style
          "sphinx_rtd_theme",
          # Render markdown in sphinx docs
          "recommonmark",
          # Render Jupyter Notebooks in sphinx docs
          "nbsphinx",
          # Render ASCII art diagram (https://aafigure.readthedocs.io)
          "aafigure",
          # Bridge between doxygen and sphinx. Used to generate C++ API docs
          "breathe",
          # Repair wheels to embed shared libraries.
          # - 3.2.0: enable defining custom patcher
          # - 3.3.0: Support Python 3.9 and manylinux_2_24 images
          # - 4.0.0: Many bug fixes, including RPATH of dependencies
          "auditwheel>=4.0.0"
      ]
    },
    zip_safe=False
)
