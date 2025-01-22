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
    python_requires=">=3.10,<3.14",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13"
    ],
    keywords="reinforcement-learning robotics gym jiminy",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=[
        f"gym-jiminy[toolbox]~={gym_jiminy_version}"
    ],
    zip_safe=False
)
