# install tool-chain library
sudo apt update
sudo apt install -y build-essential make \
                      git g++ pkg-config curl libfreetype6-dev \
                      libcanberra-gtk-module libcanberra-gtk3-module
bash cmake-install.sh

# install python3.7
sudo apt install python3.7-dev
sudo rm /usr/bin/python3
sudo ln -s /usr/bin/python3.7 /usr/bin/python3
wget https://files.pythonhosted.org/packages/b7/2d/ad02de84a4c9fd3b1958dc9fb72764de1aa2605a9d7e943837be6ad82337/pip-21.0.1.tar.gz
tar -xzvf pip-21.0.1.tar.gz
cd pip-21.0.1
python3.7 setup.py install

pip install -r requirements.txt

# build and install trt-oss
git clone https://github.com/NVDIA/TensorRT
cd TensorRT
checkout release/8.2
git submodule update --init --recursive

export TRT_OSSPATH=`pwd`
export TRT_LIBPATH=/usr/lib/aarch64-linux-gnu
export TRT_INCPATH=/usr/include/aarch64-linux-gnu

mkdir -p build && cd build
sudo cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=aarch64 -DCUDA_VERSION=10.2
CC=/usr/bin/gcc make -j$(nproc)
sudo make install
sudo cp ./out/libnvinfer_plugin* $TRT_LIBPATH

# build and install trt-pip
export EXT_PATH=~/external
mkdir -p $EXP_PATH
cd $EXT_PATH
git clone https://github.com/pybind/pybind11.git
git checkout v2.9
wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
tar zxvf python-3.7.5.tgz
cp -r python-3.7.5/Include $EXT_PATH/python3.7/include
wget http://archive.ubuntu.com/ubuntu/pool/universe/p/python3.7/libpython3.7-dev_3.7.0~b3-1_amd64.deb
ar x libpython3.7-dev_3.7.0~b3-1_amd64.deb
tar -xvf data.tar.xz
cp -r ./usr/include/arm64 $EXT_PATH/python3.7/include

cd $TRT_OSSPATH/python
sudo PYTHON_MAJOR_VERSION=3 PYTHON_MINOR_VERSION=7 TARGET_ARCHITECTURE=aarch64 TRT_OSSPATH=$TRT_OSSPATH bash ./build.sh
pip install build/dist/tensorrt-*.whl
