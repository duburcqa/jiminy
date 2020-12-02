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
    url="https://github.com/Wandercraft/jiminy",
    download_url=("https://github.com/Wandercraft/jiminy/archive/"
                  "@PROJECT_VERSION@.tar.gz"),
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
        # Standard interface library for reinforcement learning.
        # 0.17.3 introduces iterable space dict.
        "gym>=0.17.3",
        # Use internally to speedup computation of simple methods.
        # Disable automatic forward compatibility with newer versions because
        # numba relies on llvmlite, for which wheels take some time before
        # being available on Pypi, making the whole installation process fail.
        "numba<0.53",
        f"jiminy-py~={version_required}"
    ],
    extras_require=extras,
    zip_safe=False
)
