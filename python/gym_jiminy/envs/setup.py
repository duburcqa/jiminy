from setuptools import setup, find_namespace_packages


version = __import__("jiminy_py").__version__
version_required = ".".join(version.split(".")[:2])

setup(
    name="gym_jiminy_zoo",
    version=version,
    description=(
        "Classic Reinforcement learning environments for Gym Jiminy."),
    author="Alexis Duburcq",
    author_email="alexis.duburcq@wandercraft.eu",
    maintainer="Wandercraft",
    license="MIT",
    python_requires=">=3.6",
    classifiers=[
        "DDevelopment Status :: 4 - Beta",
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
    package_data={"gym_jiminy.envs": ["data/**/*"]},
    include_package_data=True,
    install_requires=[
        "gym>=0.16.0",
        "numba",
        f"jiminy-py~={version_required}"
    ],
    zip_safe=False
)
