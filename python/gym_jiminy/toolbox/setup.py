from pkg_resources import get_distribution
from setuptools import setup, find_namespace_packages


version = get_distribution('gym_jiminy_common').version

setup(
    name="gym_jiminy_toolbox",
    version=version,
    description=(
        "Reinforcement learning toolbox based on Pytorch for Gym Jiminy."),
    author="Alexis Duburcq",
    author_email="alexis.duburcq@wandercraft.eu",
    maintainer="Wandercraft",
    license="MIT",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
    keywords="reinforcement-learning robotics gym jiminy",
    packages=find_namespace_packages(),
    install_requires=[
        f"gym_jiminy_common~={version}",
        "tensorboard"
    ],
    zip_safe=False
)
