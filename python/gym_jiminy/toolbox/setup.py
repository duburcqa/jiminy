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
    python_requires=">=3.6,<3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
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
        f"gym-jiminy=={version}",
        # Used to compute convex hull.
        # No wheel is distributed for PyPy on pypi, and pip is only able to
        # build from source after install `libatlas-base-dev` system
        # dependency.
        # 1.2.0 fixes `fmin_slsqp` optimizer returning wrong `imode` output.
        # 1.8.0: `scipy.spatial.qhull._Qhull` is no longer exposed.
        "scipy>=1.2.0,<1.8.0"
    ],
    zip_safe=False
)
