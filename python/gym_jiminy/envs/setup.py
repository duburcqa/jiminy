from importlib.metadata import version
from setuptools import setup, find_namespace_packages


gym_jiminy_version = version('gym-jiminy')

setup(
    name="gym-jiminy-zoo",
    version=gym_jiminy_version,
    description=(
        "Classic Reinforcement learning environments for Gym Jiminy."),
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
    include_package_data=True,
    install_requires=[
        f"gym-jiminy[toolbox]~={gym_jiminy_version}",
        # Backport of `importlib.resources` for Python 3.7+
        # - 1.3.0: contributed to the standard library of Python 3.9
        "importlib_resources>=1.3.0;python_version<'3.9'"
    ],
    zip_safe=False
)
