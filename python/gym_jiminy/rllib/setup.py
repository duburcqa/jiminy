from importlib.metadata import version
from setuptools import setup, find_namespace_packages


gym_jiminy_version = version('gym-jiminy')

setup(
    name="gym-jiminy-rllib",
    version=gym_jiminy_version,
    description=(
        "Specialized Reinforcement learning toolbox based on Ray RLlib for "
        "Gym Jiminy."),
    author="Alexis Duburcq",
    author_email="alexis.duburcq@gmail.com",
    maintainer="Alexis Duburcq",
    license="MIT",
    python_requires=">=3.8,<3.12",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11"
    ],
    keywords="reinforcement-learning robotics gym jiminy",
    packages=find_namespace_packages(),
    package_data={"gym_jiminy.rllib": ["py.typed"]},
    install_requires=[
        f"gym-jiminy~={gym_jiminy_version}",
        # Highly efficient distributed computation library used for RL
        # - <1.6.0: GPU detection must be patched to work
        # - 1.11.0: Breaking changes
        "ray[default,rllib]~=2.2.0",
        # Used for monitoring (logging and publishing) learning progress
        "tensorboardX",
        # Plot data directly in terminal to monitor stats without X-server
        "plotext>=5.0.0"
    ],
    zip_safe=False
)
