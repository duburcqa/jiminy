from setuptools import setup, dist


# force setuptools to recognize that this is
# actually a binary distribution
class BinaryDistribution(dist.Distribution):
    def has_ext_modules(foo):
        return True


setup(name='jiminy',
      version='1.0.0',
      description="Python binging of the C++ Jiminy open-source simulator.",
      author = 'Alexis Duburcq',
      author_email = 'alexis.duburcq@wandercraft.eu',
      maintainer = 'Wandercraft',
      url='https://github.com/Wandercraft/jiminy',
      packages=['jiminy_pywrap'],
      package_data={'jiminy_pywrap': ['libjiminy_pywrap.so']},
      include_package_data=True, # make sure the shared library is included
      distclass=BinaryDistribution
)