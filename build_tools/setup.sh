# Eigen > 3.3.0 is not compatible with Boost < 1.71 because of odeint module
# The latest version of Boost available using apt-get is 1.65,
# and currently robotpkg-py36-eigenpy depends of this release.
# Conversely, the oldest version of Eigen3 available using apt-get is 3.3.4,
# and compiling 3.2.10 from source does not generate the cmake configuration
# fles required by `find_package`.

sudo apt update && \
sudo apt install -y python3-numpy python3-setuptools && \ # libeigen3-dev
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 && \
pip install wheel && \

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub bionic robotpkg' >> /etc/apt/sources.list.d/robotpkg.list" && \
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add - && \
sudo apt update && \
sudo apt install -y doxygen graphviz libboost-all-dev && \ # libeigen3-dev
sudo apt install -y robotpkg-urdfdom=0.3.0r2 robotpkg-urdfdom-headers=0.3.0 \
                    robotpkg-py36-eigenpy robotpkg-py36-pinocchio

wget https://github.com/eigenteam/eigen-git-mirror/archive/3.2.10.tar.gz && \
tar xvzf 3.2.10.tar.gz --one-top-level=eigen-3.2.10 --strip-components 1 && \
sudo mkdir /usr/include/eigen3/ && \
sudo cp -r eigen-3.2.10/Eigen /usr/include/eigen3/ && \
sudo cp -r eigen-3.2.10/unsupported /usr/include/eigen3/