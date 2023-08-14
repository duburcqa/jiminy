#!/bin/bash

# Script for installing pre-compiled binaries of the required dependencies through apt-get on Ubuntu

export DEBIAN_FRONTEND=noninteractive

# Make sure the script has root privilege
if ! [ $(id -u) = 0 ]; then
    echo 'This script must be executed using `sudo env "PATH=$PATH"`.'
    exit 1
fi

# Set SUDO_UID to 0 (root) if not defined, which may happen in docker container
if [ -z ${SUDO_UID+x} ]; then
    SUDO_UID=0;
fi

# Prefix commands with sudo if necessary
if type "sudo" > /dev/null 2>&1; then
    SUDO_CMD="sudo -u $(id -nu ${SUDO_UID}) env 'PATH=${PATH}'"
fi

# Determine if the script is being executed on Ubuntu
if [ -f /etc/lsb-release ]; then
    source /etc/lsb-release
    if [ "${DISTRIB_ID}" != "Ubuntu" ] ; then
        echo "Not running on Ubuntu. Aborting..."
        exit 1
    fi
else
    echo "Not running on Ubuntu. Aborting..."
    exit 1
fi
echo "-- Linux distribution: ${DISTRIB_ID} ${DISTRIB_CODENAME}"

# Check if current python executable has the same version as the built-in one
GET_PYTHON_VERSION="python3 -c \"import sys ; print('.'.join(map(str, sys.version_info[:2])), end='')\""
PYTHON_VERSION="$(eval ${GET_PYTHON_VERSION})"
if type "sudo" > /dev/null 2>&1; then
    PYTHON_SYS_VERSION="$(sudo -s eval ${GET_PYTHON_VERSION})"
    if [ "${PYTHON_VERSION}" != "${PYTHON_SYS_VERSION}" ]; then
        echo "Python version must match the built-in one if using a virtual env."
        exit 1
    fi
fi
echo "-- Python executable: $(which python3)"
echo "-- Python version: ${PYTHON_VERSION}"

# Get Python 3 executable and wrtiable site packages location
PYTHON_BIN="python${PYTHON_VERSION}"
PYTHON_SITELIB="$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'), end='')")"
echo "-- Python default site-packages: ${PYTHON_SITELIB}"
if ! ${SUDO_CMD} test -w "${PYTHON_SITELIB}" ; then
    PYTHON_SITELIB="$(${SUDO_CMD} python3 -m site --user-site)"
fi
echo "-- Python writable site-packages: ${PYTHON_SITELIB}"

# Install Python 3 standard utilities
apt update && \
apt install -y python3-pip && \
${SUDO_CMD} python3 -m pip install setuptools wheel "pip>=20.3" && \
${SUDO_CMD} python3 -m pip install "numpy>=1.21" "numba>=0.54.0"

# Install standard linux utilities
apt install -y gnupg curl wget build-essential cmake doxygen graphviz pandoc

# Install some additional dependencies
apt install -y libeigen3-dev libboost-all-dev liboctomap-dev

# Install OpenGL
apt install -y mesa-utils

# Add robotpkg apt repository if necessary
if ! grep -q "^deb .*robotpkg.openrobots.org" /etc/apt/sources.list.d/*; then
    sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub ${DISTRIB_CODENAME} robotpkg' >> /etc/apt/sources.list.d/robotpkg.list" && \
    curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | apt-key add - && \
    apt update
fi

# Install robotpkg tools suite.
# Note that `apt-get` is used instead of `apt` because it supports wildcard in package names
apt-mark unhold "robotpkg-py3*-eigenpy" "robotpkg-py3*-hpp-fcl" "robotpkg-py3*-pinocchio"
apt-get install -y --allow-downgrades --allow-unauthenticated \
    robotpkg-urdfdom-headers=1.0.4 robotpkg-hpp-fcl=2.3.5 robotpkg-pinocchio=2.6.20 \
    robotpkg-py3*-eigenpy=3.1.1 robotpkg-py3*-hpp-fcl=2.3.5 robotpkg-py3*-pinocchio=2.6.20
apt-mark hold "robotpkg-py3*-eigenpy" "robotpkg-py3*-hpp-fcl" "robotpkg-py3*-pinocchio"

# Add openrobots libraries to python packages search path
if ! [ -f "${PYTHON_SITELIB}/openrobots.pth" ]; then
    ${SUDO_CMD} bash -c " \
    echo 'export LD_LIBRARY_PATH=\"/opt/openrobots/lib:\${LD_LIBRARY_PATH}\"' >> \${HOME}/.bashrc && \
    echo 'export PATH=\"\${PATH}:/opt/openrobots/bin:/root/.local/bin\"' >> \${HOME}/.bashrc && \
    mkdir -p '${PYTHON_SITELIB}' && \
    touch '${PYTHON_SITELIB}/openrobots.pth' && \
    echo '/opt/openrobots/lib/${PYTHON_BIN}/site-packages/' > '${PYTHON_SITELIB}/openrobots.pth'"
fi
