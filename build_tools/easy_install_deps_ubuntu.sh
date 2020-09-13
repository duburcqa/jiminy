#!/bin/bash

# Script for installing pre-compiled binaries of the required dependencies through apt-get on Ubuntu

# Eigen > 3.3.0 is not compatible with Boost < 1.71 because of odeint module
# The latest version of Boost available using apt-get is 1.65 on Ubuntu 18,
# and currently robotpkg-py36-eigenpy depends of this release.
# Conversely, the oldest version of Eigen3 available using apt-get is 3.3.4,
# and compiling 3.2.10 from source does not generate the cmake configuration
# files required by `find_package`.
# There is no such issue on Ubuntu 20 since it ships with Boost 1.71.0.

export DEBIAN_FRONTEND=noninteractive

# Determine if the script is being executed on Ubuntu
if [ -f /etc/lsb-release ]; then
    source /etc/lsb-release
    if [ $DISTRIB_ID != "Ubuntu" ] || ( [ $DISTRIB_RELEASE != "18.04" ] && [ $DISTRIB_RELEASE != "20.04" ] ) ; then
        echo "Not running on Ubuntu 18 or 20. Aborting..."
        exit 0
    fi
else
    echo "Not running on Ubuntu 18 or 20. Aborting..."
    exit 0
fi

# Get Python 3 executable
PYTHON_BIN="$(basename $(readlink $(which python3)))"

# Install Python 3 tools
apt update && \
apt install -y sudo python3-setuptools python3-pip python3-tk && \
sudo -u $(id -nu $SUDO_UID) python3 -m pip install --upgrade pip && \
sudo -u $(id -nu $SUDO_UID) python3 -m pip install wheel && \
sudo -u $(id -nu $SUDO_UID) python3 -m pip install numpy

# Install standard linux utilities
apt install -y gnupg curl wget build-essential cmake doxygen graphviz

# Install old Eigen3 version
if [ $DISTRIB_RELEASE == "18.04" ] ; then
    if ! [ -d "/usr/include/eigen3.2.10/" ] ; then
        wget https://github.com/eigenteam/eigen-git-mirror/archive/3.2.10.tar.gz && \
        tar xzf 3.2.10.tar.gz --one-top-level=eigen-3.2.10 --strip-components 1 && \
        mkdir /usr/include/eigen3.2.10/ && \
        cp -r eigen-3.2.10/Eigen /usr/include/eigen3.2.10/ && \
        cp -r eigen-3.2.10/unsupported /usr/include/eigen3.2.10/ && \
        rm -r eigen-3.2.10 3.2.10.tar.gz
    fi
else
    apt install -y libeigen3-dev
fi

# Install some additional dependencies
apt install -y libboost-all-dev liboctomap-dev

# Install robotpkg tools suite
if ! [-d "/opt/openrobots/lib/${PYTHON_BIN}/site-packages/" ] ; then
    sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub ${DISTRIB_CODENAME} robotpkg' >> /etc/apt/sources.list.d/robotpkg.list" && \
    curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | apt-key add - && \
    apt update

    # apt-get must be used instead of apt to support wildcard in package name on Ubuntu 20
    apt-get install -y --allow-downgrades --allow-unauthenticated \
        robotpkg-urdfdom=1.0.3 robotpkg-urdfdom-headers=1.0.4 robotpkg-hpp-fcl=1.4.5 \
        robotpkg-py3*-qt5-gepetto-viewer=4.9.0r3 robotpkg-py3*-qt5-gepetto-viewer-corba=5.4.0r2 robotpkg-py3*-omniorbpy=4.2.4 \
        robotpkg-py3*-eigenpy=2.5.0 robotpkg-py3*-hpp-fcl=1.4.5 \
        robotpkg-pinocchio=2.4.7 robotpkg-py3*-pinocchio=2.4.7

    sudo -H -u $(id -nu $SUDO_UID) bash -c " \
    echo 'export LD_LIBRARY_PATH=\"/opt/openrobots/lib:\${LD_LIBRARY_PATH}\"' >> \$HOME/.bashrc && \
    echo 'export PATH=\"\${PATH}:/opt/openrobots/bin\"' >> \$HOME/.bashrc && \
    mkdir -p \$HOME/.local/lib/${PYTHON_BIN}/site-packages && \
    touch \$HOME/.local/lib/${PYTHON_BIN}/site-packages/openrobots.pth && \
    echo /opt/openrobots/lib/${PYTHON_BIN}/site-packages/ > \$HOME/.local/lib/${PYTHON_BIN}/site-packages/openrobots.pth"
fi
