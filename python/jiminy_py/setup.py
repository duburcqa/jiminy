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
      package_data = {'jiminy_py': ['**/*.dll', '**/*.so', '**/*.pyd', '**/*.html', '**/*.js']},
      entry_points={'console_scripts': [
          'jiminy_plot=jiminy_py.log:plot_log',
          'jiminy_meshcat_server=jiminy_py.viewer:start_zmq_server_standalone'
      ]},
      include_package_data = True,  # make sure the shared library is included
      distclass = BinaryDistribution,
      cmdclass = {'install': InstallPlatlib},
      install_requires = [
          'wheel',
          'xmltodict',
          'pillow',
          'tqdm',
          'numpy',
          'scipy',
          'matplotlib',
          'meshcat @ git+https://github.com/rdeits/meshcat-python.git',  # Waiting for release 0.19
          'psutil'
      ],
      extras_require = {
        'record': [
            'requests_html', 
            'opencv-python-headless'
        ]
      }
)
