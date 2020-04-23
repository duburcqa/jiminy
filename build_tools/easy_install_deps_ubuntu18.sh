#!/bin/bash

# Script for Ubuntu 18 installing pre-compiled binaries of the required dependencies through apt-get

# Eigen > 3.3.0 is not compatible with Boost < 1.71 because of odeint module
# The latest version of Boost available using apt-get is 1.65, and currently
# robotpkg-py36-eigenpy depends of this release.
# Conversely, the oldest version of Eigen3 available using apt-get is 3.3.4,
# and compiling 3.2.10 from source does not generate the cmake configuration
# files required by `find_package`.

export DEBIAN_FRONTEND=noninteractive

# Install Python 3 tools
apt update && \
apt install -y sudo python3-setuptools python3-pip python3-tk && \
update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
sudo -u $(id -nu $SUDO_UID) pip install wheel && \
sudo -u $(id -nu $SUDO_UID) pip install numpy

# Install standard linux utilities and boost tools suite
apt install -y gnupg curl wget build-essential cmake doxygen graphviz libboost-all-dev # libeigen3-dev

# Install Eigen
if ! [ -d "/usr/include/eigen3/" ] ; then
    wget https://github.com/eigenteam/eigen-git-mirror/archive/3.2.10.tar.gz && \
    tar xzf 3.2.10.tar.gz --one-top-level=eigen-3.2.10 --strip-components 1 && \
    mkdir /usr/include/eigen3/ && \
    cp -r eigen-3.2.10/Eigen /usr/include/eigen3/ && \
    cp -r eigen-3.2.10/unsupported /usr/include/eigen3/ && \
    rm -r eigen-3.2.10 3.2.10.tar.gz
fi

# Install robotpkg tools suite
if ! [-d "/opt/openrobots/lib/python3.6/site-packages/" ] ; then
    sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub bionic robotpkg' >> /etc/apt/sources.list.d/robotpkg.list" && \
    curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | apt-key add - && \
    apt update && \
    apt install -y robotpkg-urdfdom=0.3.0r2 robotpkg-urdfdom-headers=0.3.0 robotpkg-hpp-fcl=1.3.0 robotpkg-py36-hpp-fcl=1.3.0 \
                   robotpkg-pinocchio=2.2.2 robotpkg-py36-eigenpy=2.0.2 robotpkg-py36-pinocchio=2.2.2 && \
    echo 'export LD_LIBRARY_PATH="/opt/openrobots/lib"' >> $HOME/.bashrc && \
    sudo -u $(id -nu $SUDO_UID) mkdir -p $HOME/.local/lib/python3.6/site-packages && \
    sudo -u $(id -nu $SUDO_UID) touch $HOME/.local/lib/python3.6/site-packages/openrobots.pth && \
    echo "/opt/openrobots/lib/python3.6/site-packages/" > $HOME/.local/lib/python3.6/site-packages/openrobots.pth
fi
