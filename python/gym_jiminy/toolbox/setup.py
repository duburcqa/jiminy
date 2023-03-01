from pkg_resources import get_distribution
from setuptools import setup, find_namespace_packages


version = get_distribution('gym-jiminy').version

setup(
    name="gym-jiminy-toolbox",
    version=version,
    description=(
        "Generic Reinforcement learning toolbox based on Pytorch for Gym "
        "Jiminy."),
    author="Alexis Duburcq",
    author_email="alexis.duburcq@gmail.com",
    maintainer="Alexis Duburcq",
    license="MIT",
    python_requires=">=3.8,<3.12",
    classifiers=[
        "Development Status :: 3 - Alpha",
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
    install_requires=[
        f"gym-jiminy=={version}",
        # Used to compute convex hull.
        # No wheel is distributed on pypi for PyPy, and pip requires to install
        # `libatlas-base-dev` system dependency to build it from source.
        # 1.8.0: `scipy.spatial.qhull` low-level API changes.
        "scipy>1.8.0"
    ],
    zip_safe=False
)
