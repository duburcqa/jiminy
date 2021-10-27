from pkg_resources import get_distribution
from setuptools import setup, find_namespace_packages


version = get_distribution('gym_jiminy').version

setup(
    name="gym_jiminy_zoo",
    version=version,
    description=(
        "Classic Reinforcement learning environments for Gym Jiminy."),
    author="Alexis Duburcq",
    author_email="alexis.duburcq@gmail.com",
    maintainer="Alexis Duburcq",
    license="MIT",
    python_requires=">=3.6,<3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ],
    keywords="reinforcement-learning robotics gym jiminy",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=[
        f"gym_jiminy=={version}"
    ],
    zip_safe=False
)
