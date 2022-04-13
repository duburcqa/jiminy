from pkg_resources import get_distribution
from setuptools import setup, find_namespace_packages


version = get_distribution('gym_jiminy').version

setup(
    name="gym_jiminy_rllib",
    version=version,
    description=(
        "Specialized Reinforcement learning toolbox based on Ray RLlib for "
        "Gym Jiminy."),
    author="Alexis Duburcq",
    author_email="alexis.duburcq@gmail.com",
    maintainer="Alexis Duburcq",
    license="MIT",
    python_requires=">=3.6,<3.11",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10"
    ],
    keywords="reinforcement-learning robotics gym jiminy",
    packages=find_namespace_packages(),
    install_requires=[
        f"gym_jiminy[toolbox]=={version}",
        # - <1.6.0: GPU detection must be patched to work
        # - >=1.6.0: Cannot load checkpoints generated by Python < 3.8 using
        # Python >= 3.8.
        # - 1.9.0: Breaking changes
        # - 1.10.0: Breaking changes
        # - 1.11.0: Breaking changes
        "ray[default,rllib]>=1.10.0,<1.11.0",
        # Used for logging
        "tensorboardX",
        # Plot data directly in terminal to monitor stats without X-server
        "plotext>=5.0.0"
    ],
    zip_safe=False
)
