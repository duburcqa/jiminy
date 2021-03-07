from setuptools import setup, dist, find_packages
from setuptools.command.install import install


# Force setuptools to recognize that this is actually a binary distribution
class BinaryDistribution(dist.Distribution):
    def is_pure(self) -> bool:
        return False

    def has_ext_modules(self) -> bool:
        return True


# Force setuptools to not consider shared libraries as purelib
class InstallPlatlib(install):
    def finalize_options(self) -> None:
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


setup(
    name="jiminy_py",
    version="@PROJECT_VERSION@",
    description=("Fast and light weight simulator of rigid poly-articulated "
                 "systems."),
    long_description=open("@SOURCE_DIR@/README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/duburcqa/jiminy",
    download_url=("https://github.com/duburcqa/jiminy/archive/"
                  "@PROJECT_VERSION@.tar.gz"),
    author="Alexis Duburcq",
    author_email="alexis.duburcq@gmail.com",
    maintainer="Alexis Duburcq",
    license="MIT",
    python_requires=">=3.6,<3.10",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
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
    package_data={"jiminy_py": [
        "**/*.dll", "**/*.so", "**/*.pyd", "**/*.html", "**/*.js"
    ]},
    include_package_data=True,
    entry_points={"console_scripts": [
        "jiminy_plot=jiminy_py.log:plot_log",
        ("jiminy_meshcat_server="
         "jiminy_py.meshcat.server:start_meshcat_server_standalone")
    ]},
    install_requires=[
        # Used internally by Viewer to read/write snapshots.
        # >= 8.0 is required to support Python3.9.
        "pillow>=8.0",
        # Add support of TypedDict to any Python 3 version.
        "typing_extensions",
        # Display elegant and versatile process bar.
        "tqdm",
        # Standard library for matrix algebra.
        "numpy",
        # Used internally for interpolation and filtering.
        "scipy",
        # Standard library to generate figures.
        # 3.3 is broken on Windows 64 bits and cannot be installed properly.
        "matplotlib<3.3",
        # Used internally to read HDF5 format log files.
        "h5py",
        # Used internally by Robot to replace meshes by associated minimal
        # volume bounding box.
        "trimesh",
        # Parser for Jiminy's hardware description file.
        "toml",
        # Web-based mesh visualizer used as Viewer's backend.
        # 0.19.0 introduces many new features, including loading generic
        # geometries and jiminy_py viewer releases on it for rendering
        # collision bodies.
        "meshcat>=0.0.19",
        # Standalone mesh visualizer used as Viewer's backend.
        "panda3d_viewer",
        # Used internally by Viewer to record video programmatically when
        # Meshcat is not used as rendering backend.
        # Python3.9 is supported since version 4.4.0.46.
        "opencv-python-headless>=4.4.0.46",
        # Used internally by Viewer to detect running Meshcat servers and avoid
        # orphan child processes.
        "psutil",
        # Used internally by Viewer to enable recording video programmatically
        # while using Meshcat as backend.
        "pyppeteer",
        # Used internally by Viewer to send/receive Javascript requests while
        # recording video using Meshcat backend.
        "requests_html"
    ],
    extras_require={
      "dev": [
          # Check PEP8 conformance of Python native code
          "flake8",
          # Python linter
          "pylint",
          # Python static type checker
          "mypy",
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
          "breathe"
      ]
    },
    zip_safe=False
)
