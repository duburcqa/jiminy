from setuptools import setup, find_packages

setup(name = 'jiminy_py',
      version = '1.0',
      description = 'Package containing python-native helper methods for Jiminy Open Source.',
      author = 'Wandercraft',
      maintainer = 'Alexis Duburcq',
      maintainer_email='alexis.duburcq@wandercraft.eu',
      url='https://github.com/Wandercraft/jiminy',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires = ['meshcat'],
      zip_safe = False
)
