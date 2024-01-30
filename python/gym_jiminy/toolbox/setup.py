from importlib.metadata import version
from setuptools import setup, find_namespace_packages


gym_jiminy_version = version('gym-jiminy')

setup(
    name="gym-jiminy-toolbox",
    version=gym_jiminy_version,
    description=(
        "Generic Reinforcement learning toolbox based on Pytorch for Gym "
        "Jiminy."),
    author="Alexis Duburcq",
    author_email="alexis.duburcq@gmail.com",
    maintainer="Alexis Duburcq",
    license="MIT",
    python_requires=">=3.8,<3.13",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ],
    keywords="reinforcement-learning robotics gym jiminy",
    packages=find_namespace_packages(),
    package_data={"gym_jiminy.toolbox": ["py.typed"]},
    install_requires=[
        f"gym-jiminy~={gym_jiminy_version}",
        # Used to compute convex hull.
        # No wheel is distributed on pypi for PyPy, and pip requires to install
        # `libatlas-base-dev` system dependency to build it from source.
        # 1.8.0: `scipy.spatial.qhull` low-level API changes.
        # 1.9.2: First release to support Python 3.11
        "scipy>=1.9.2"
    ],
    zip_safe=False
)
