from setuptools import setup, find_packages


version = __import__('jiminy_py').__version__
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
            'stable_baselines',
            'jiminy-py~=1.2'
      ]
)
