from setuptools import setup, find_packages

setup(name = 'gym_jiminy',
      version = '1.0.4',
      description = 'Python-native interface between OpenAI Gym and Jiminy open-source simulator.',
      author = 'Alexis Duburcq',
      author_email = 'alexis.duburcq@wandercraft.eu',
      maintainer = 'Wandercraft',
      url='https://github.com/Wandercraft/jiminy',
      packages=find_packages('.'),
      package_data = {'gym_jiminy.envs': ['data/**/*']},
      include_package_data = True, # make sure the data folder is included
      install_requires = [
            'gym',
            'stable_baselines',
            'jiminy-py==1.0.4'
            ]
)