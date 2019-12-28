
##### Prerequisites

```bash
sudo apt install -y libeigen3-dev doxygen libtinyxml-dev cmake git
```

##### Custom Boost version

If for some reasons you need a specific version of Boost, use the following installation procedure:

```bash
wget http://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.bz2 && \
sudo tar --bzip2 -xf boost_1_65_1.tar.bz2 && \
cd boost_1_65_1 && sudo ./bootstrap.sh && \
sudo ./b2  --address-model=64 --architecture=x86 -j8 install && \
cd .. && rm boost_1_65_1.tar.bz2
```

If using Conda venv for Python, do not forget to update Python location in `boost_1_65_1/project-config.jam`. For instance:

```bash
import python ;
if ! [ python.configured ]
{
    using python : 3.5 : /home/aduburcq/anaconda3/envs/py35 : /home/aduburcq/anaconda3/envs/py35/include/python3.5m/ ;
}
```

At least for Boost 1.65.1, it is necessary to [fix Boost Numpy](https://github.com/boostorg/python/pull/218/commits/0fce0e589353d772ceda4d493b147138406b22fd). Update `wrap_import_array` method in `boost_1_65_1/libs/python/src/numpy/numpy.cpp`:
```bash
static void * wrap_import_array()
{
  import_array();
  return NULL;
}
```

Finally, add the following lines in your `.bashrc`:

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

In addition, make sure to be using `numpy>=1.16.0`.

##### Install Eigenpy

```bash
git clone --recursive https://github.com/stack-of-tasks/eigenpy.git && \
cd eigenpy && mkdir -p build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DPYTHON_EXECUTABLE:FILEPATH=/home/aduburcq/anaconda3/envs/py35/bin/python \
         -DBOOST_ROOT:PATHNAME=/usr/local/ && \
sudo make install -j8 && \
cd .. && rm -f eigenpy
```

##### Install Urdfdom 0.3.0 (soversion 0.2.0)

```bash
git clone git://github.com/ros/console_bridge.git && \
cd console_bridge && git checkout 0.2.7 && \
mkdir -p build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/usr/local && \
sudo make install -j8 && \
cd .. && rm -f console_bridge
```

```bash
git clone git://github.com/ros/urdfdom_headers.git && \
cd urdfdom_headers && git checkout 0.3.0 && \
mkdir -p build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/usr/local && \
sudo make install -j8 && \
cd .. && rm -f urdfdom_headers
```

```bash
git clone git://github.com/ros/urdfdom.git && \
cd urdfdom && git checkout 0.3.0 && \
mkdir -p build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/usr/local && \
sudo make install -j8 && \
cd .. && rm -f urdfdom
```

At least for Urdfdom 0.3.0, it is necessary to fix Python Array API initialization and code Python 2.7 only:
update `DuckTypedFactory` method in `urdf_parser_py/src/urdf_parser_py/xml_reflection/core.py` to replace `except Exception, e` by `except Exception as e:`, and
`install` call in `urdf_parser_py/CMakeLists.txt` to remove `--install-layout deb`.

##### Install Pinocchio v2.1.10 (lower versions are not compiling properly)

```bash
git clone --recursive https://github.com/stack-of-tasks/pinocchio && \
cd pinocchio && git checkout v2.1.10 && \
mkdir -p build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DPYTHON_EXECUTABLE:FILEPATH=/home/aduburcq/anaconda3/envs/py35/bin/python \
         -DBOOST_ROOT:PATHNAME=/usr/local/ \
         -DPROJECT_URL="http://github.com/stack-of-tasks/pinocchio" && \
sudo make install -j8 && \
cd .. && rm -f pinocchio
```

##### Configure .bashrc

Add the following lines:

```bash
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/lib/pythonXXX/site-packages:$PYTHONPATH
```