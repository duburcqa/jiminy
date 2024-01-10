from itertools import chain
from importlib.metadata import version
from setuptools import setup, find_namespace_packages


jiminy_version = version('jiminy-py')

extras = {
    "zoo": [
        f"gym-jiminy-zoo~={jiminy_version}",
    ],
    "toolbox": [
        f"gym-jiminy-toolbox~={jiminy_version}"
    ],
    "rllib": [
        f"gym-jiminy-rllib~={jiminy_version}"
    ]
}
extras["all"] = list(set(chain.from_iterable(extras.values())))

setup(
    name="gym-jiminy",
    version=jiminy_version,
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
    package_data={"gym_jiminy.common": ["py.typed"]},
    install_requires=[
        f"jiminy-py~={jiminy_version}",
        # Use to perform linear algebra computation
        "numpy",
        # Use internally to speedup computation of math methods
        # - 0.54: Adds 'np.clip'
        "numba>=0.54.0",
        # Use to operate on nested data structure conveniently.
        # - 0.1.7 breaking API and internal changes.
        "dm-tree>=0.1.7",
        # Standard interface library for reinforcement learning.
        # - `gym` has been replaced by `gymnasium` for 0.26.0+
        # - 0.28.0: fully typed
        # - bound version for resilience to recurrent API breakage
        "gymnasium>=0.26,<0.29",
        # For backward compatibility of latest Python typing features
        # - TypeAlias has been added with Python 3.10
        "typing_extensions"
    ],
    extras_require=extras,
    zip_safe=False
)
