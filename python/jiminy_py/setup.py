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

setup(name = 'jiminy_py',
      version = '@PROJECT_VERSION@',
      license = 'MIT',
      description = 'Python-native helper methods and wrapping classes for Jiminy open-source simulator.',
      author = 'Alexis Duburcq',
      author_email = 'alexis.duburcq@wandercraft.eu',
      maintainer = 'Wandercraft',
      url = 'https://github.com/Wandercraft/jiminy',
      download_url = 'https://github.com/Wandercraft/jiminy/archive/@PROJECT_VERSION@.tar.gz',
      packages = find_packages('src'),
      package_dir = {'': 'src'},
      package_data = {'jiminy_py': [
          '**/*.dll', '**/*.so', '**/*.pyd', '**/*.html', '**/*.js'
      ]},
      entry_points = {'console_scripts': [
          'jiminy_plot=jiminy_py.log:plot_log',
          'jiminy_meshcat_server=jiminy_py.meshcat.server:start_meshcat_server_standalone'
      ]},
      include_package_data = True,  # make sure the shared library is included
      distclass = BinaryDistribution,
      cmdclass = {'install': InstallPlatlib},
      install_requires = [
          'pillow',           # Used internal by the viewer to read/write snapshots
          'tqdm',             # Used to display elegant and versatile process bar
          'numpy',            # Standard library for matrix algebra
          'scipy',            # Used internally for interpolation and filtering
          'matplotlib',       # Standard library for showing figures
          'toml',             # Parser for Jiminy-specific robot sensor description files
          'meshcat>=0.0.18',  # Web-based mesh visualizer used by default as viewer's backend
          'psutil',           # Used internally to detect running meshcat servers and to avoid orphan child processes
          'requests_html'     # Used internally to handle viewer recording
      ],
      extras_require = {
        'gepetto': [
            'opencv-python-headless<=4.3.0.36'  # Used by the viewer to record video while using Gepetto-viewer as backend
        ]
      }
)
