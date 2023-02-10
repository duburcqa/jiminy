from itertools import chain
from pkg_resources import get_distribution
from setuptools import setup, find_namespace_packages


version = get_distribution('jiminy-py').version

extras = {
    "zoo": [
        f"gym-jiminy-zoo=={version}",
    ],
    "toolbox": [
        f"gym-jiminy-toolbox=={version}"
    ],
    "rllib": [
        f"gym-jiminy-rllib=={version}"
    ]
}
extras["all"] = list(set(chain.from_iterable(extras.values())))

setup(
    name="gym-jiminy",
    version=version,
    description=(
        "Python-native OpenAI Gym interface between Jiminy open-source "
        "simulator and Reinforcement Learning frameworks."),
    author="Alexis Duburcq",
    author_email="alexis.duburcq@gmail.com",
    url="https://github.com/duburcqa/jiminy",
    download_url=("https://github.com/duburcqa/jiminy/archive/"
                  "@PROJECT_VERSION@.tar.gz"),
    maintainer="Alexis Duburcq",
    license="MIT",
    python_requires=">=3.8,<3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10"
    ],
    keywords="reinforcement-learning robotics gym jiminy",
    packages=find_namespace_packages(),
    install_requires=[
        f"jiminy-py=={version}",
        # Use to perform linear algebra computation
        "numpy",
        # Use internally to speedup computation of math methods
        "numba",
        # Use to operate on nested data structure conveniently.
        # - 0.1.7 breaking API and internal changes.
        "dm-tree>=0.1.7",
        # Standard interface library for reinforcement learning.
        # - 0.17.3 introduces iterable space dict
        # - 0.18.0: dtype handling of flatten space
        # - >=0.18.0,<0.18.3 requires Pillow<8.0 to work, not compatible with
        #   Python 3.9.
        # - >= 0.22.0 advanced typing
        "gym>=0.18.3,<0.24.0"
    ],
    extras_require=extras,
    zip_safe=False
)
