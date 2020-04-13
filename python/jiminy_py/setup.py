from setuptools import setup, dist, find_packages


# force setuptools to recognize that this is
# actually a binary distribution
class BinaryDistribution(dist.Distribution):
    def has_ext_modules(foo):
        return True


setup(name = 'jiminy_py',
      version = '1.2.15',
      license = 'MIT',
      description = 'Python-native helper methods and wrapping classes for Jiminy open-source simulator.',
      author = 'Alexis Duburcq',
      author_email = 'alexis.duburcq@wandercraft.eu',
      maintainer = 'Wandercraft',
      url = 'https://github.com/Wandercraft/jiminy',
      download_url = 'https://github.com/Wandercraft/jiminy/archive/1.2.15.tar.gz',
      packages = find_packages('src'),
      package_dir = {'': 'src'},
      package_data = {'jiminy_py': ['**/*.dll', '**/*.so', '**/*.pyd']},
      include_package_data = True, # make sure the shared library is included
      distclass = BinaryDistribution,
      install_requires = [
          'Pillow',
          'meshcat',
          'scipy',
          'matplotlib',
          'tqdm'
      ]
)
