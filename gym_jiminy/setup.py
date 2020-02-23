from setuptools import setup

setup(name = 'gym_jiminy',
      version = '1.0.0',
      description = 'Python-native interface between OpenAI Gym and Jiminy open-source simulator.',
      author = 'Alexis Duburcq',
      author_email = 'alexis.duburcq@wandercraft.eu',
      maintainer = 'Wandercraft',
      url='https://github.com/Wandercraft/jiminy',
      install_requires = ['gym', 'jiminy-py==1.0.0']
)