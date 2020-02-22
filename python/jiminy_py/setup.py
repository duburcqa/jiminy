from setuptools import setup, find_packages

setup(name = 'jiminy_py',
      version = '1.0.0',
      description = 'Python-native helper methods and wrapping classes for Jiminy open-source simulator.',
      author = 'Alexis Duburcq',
      author_email = 'alexis.duburcq@wandercraft.eu',
      maintainer = 'Wandercraft',
      url='https://github.com/Wandercraft/jiminy',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires = ['meshcat'],
      zip_safe = False
)
