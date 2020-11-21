from itertools import chain
from setuptools import setup, find_namespace_packages


version = __import__("jiminy_py").__version__
version_required = ".".join(version.split(".")[:2])

extras = {
    "zoo": [
        f"gym_jiminy_zoo~={version_required}",
    ],
    "toolbox": [
        f"gym_jiminy_toolbox~={version_required}"
    ]
}
extras["all"] = list(set(chain.from_iterable(extras.values())))


setup(
    name="gym_jiminy",
    version=version,
    description=(
        "Python-native OpenAI Gym interface between Jiminy open-source "
        "simulator and Reinforcement Learning frameworks."),
    author="Alexis Duburcq",
    author_email="alexis.duburcq@wandercraft.eu",
    maintainer="Wandercraft",
    license="MIT",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "gym>=0.16.0",
        "numba",
        "typing_extensions",
        f"jiminy-py~={version_required}"
    ],
    extras_require=extras,
    zip_safe=False
)
