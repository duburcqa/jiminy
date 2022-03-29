from itertools import chain
from pkg_resources import get_distribution
from setuptools import setup, find_namespace_packages


version = get_distribution('jiminy_py').version

extras = {
    "zoo": [
        f"gym_jiminy_zoo=={version}",
    ],
    "toolbox": [
        f"gym_jiminy_toolbox=={version}"
    ],
    "rllib": [
        f"gym_jiminy_rllib=={version}"
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
    author_email="alexis.duburcq@gmail.com",
    url="https://github.com/duburcqa/jiminy",
    download_url=("https://github.com/duburcqa/jiminy/archive/"
                  "@PROJECT_VERSION@.tar.gz"),
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
    install_requires=[
        f"jiminy-py=={version}",
        # Use to perform linear algebra computation.
        # 1.16 introduces new array function dispatcher which had significant
        # overhead if not handle carefully.
        "numpy>=1.16",
        # Use internally to speedup computation of math methods.
        # Disable automatic forward compatibility with newer versions because
        # numba relies on llvmlite, for which wheels take some time before
        # being available on Pypi, making the whole installation process fail.
        # >=0.53 is required to support Python 3.9.
        # >=0.54 does not support Python 3.6 anymore.
        "numba",
        # Standard interface library for reinforcement learning.
        # - 0.17.3 introduces iterable space dict
        # - 0.18.0: dtype handling of flatten space
        # - >=0.18.0,<0.18.3 requires Pillow<8.0 to work, not compatible with
        #   Python 3.9.
        "gym>=0.18.3,<0.24.0"
    ],
    extras_require=extras,
    zip_safe=False
)
