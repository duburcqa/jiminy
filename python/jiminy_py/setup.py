from setuptools import setup, dist, find_packages
from setuptools.command.install import install


# Force setuptools to recognize that this is actually a binary distribution
class BinaryDistribution(dist.Distribution):
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True

# Force setuptools to not consider shared libraries as purelib
class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib

setup(name="jiminy_py",
      version="@PROJECT_VERSION@",
      description="Fast and light weight simulator of poly-articulated systems.",
      long_description=open("@SOURCE_DIR@/README.md", encoding="utf8").read(),
      long_description_content_type="text/markdown",
      url="https://github.com/Wandercraft/jiminy",
      download_url="https://github.com/Wandercraft/jiminy/archive/@PROJECT_VERSION@.tar.gz",
      author="Alexis Duburcq",
      author_email="alexis.duburcq@wandercraft.eu",
      maintainer="Wandercraft",
      license="MIT",
      python_requires=">3.6",
      classifiers=[
          "Development Status :: 4 - Stable",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8"
      ],
      keywords="robotics physics simulator",
      distclass = BinaryDistribution,
      cmdclass = {"install": InstallPlatlib},
      packages=find_packages("src"),
      package_dir={"": "src"},
      package_data={"jiminy_py": [
          "**/*.dll", "**/*.so", "**/*.pyd", "**/*.html", "**/*.js"
      ]},
      include_package_data=True,
      entry_points={"console_scripts": [
          "jiminy_plot=jiminy_py.log:plot_log",
          "jiminy_meshcat_server=jiminy_py.meshcat.server:start_meshcat_server_standalone"
      ]},
      install_requires = [
          "pillow",           # Used internal by the viewer to read/write snapshots
          "tqdm",             # Used to display elegant and versatile process bar
          "numpy",            # Standard library for matrix algebra
          "scipy",            # Used internally for interpolation and filtering
          "matplotlib<3.3",   # Standard library for showing figures
          "trimesh",          # Used internally to compute the minimal volume bounding box associated with a mesh
          "toml",             # Parser for Jiminy-specific robot sensor description files
          "meshcat>=0.0.18",  # Web-based mesh visualizer used by default as viewer's backend
          "psutil",           # Used internally to detect running meshcat servers and to avoid orphan child processes
          "requests_html"     # Used internally to handle viewer recording
      ],
      extras_require = {
        "gepetto": [
            "opencv-python-headless<=4.3.0.36"  # Used by the viewer to record video while using Gepetto-viewer as backend
        ]
      }
)
