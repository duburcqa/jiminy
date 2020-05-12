from setuptools import setup, find_packages


version = __import__('jiminy_py').__version__
version_required = '.'.join(version.split('.')[:2])
setup(name = 'gym_jiminy',
      version = version,
      license = 'MIT',
      description = 'Python-native interface between OpenAI Gym and Jiminy open-source simulator.',
      author = 'Alexis Duburcq',
      author_email = 'alexis.duburcq@wandercraft.eu',
      maintainer = 'Wandercraft',
      url = 'https://github.com/Wandercraft/jiminy',
      download_url = f'https://github.com/Wandercraft/jiminy/archive/{version}.tar.gz',
      packages = find_packages('.'),
      package_data = {'gym_jiminy.envs': ['data/**/*']},
      include_package_data = True, # make sure the data folder is included
      install_requires = [
            'gym',
            f'jiminy-py~={version_required}'
      ],
      python_requires='>3.6'
)

