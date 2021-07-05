#sudo apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    git \
#    curl \
#    vim \
#    ca-certificates \
#    libjpeg-dev \
#    libpng-dev \
#    libglfw3-dev \
#    libglm-dev \
#    libx11-dev \
#    libomp-dev \
#    libegl1-mesa-dev \
#    libsndfile1 \
#    pkg-config \
#    wget \
#    zip \
#    unzip &&\
#    rm -rf /var/lib/apt/lists/*


# Install cmake

#wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
#mkdir /opt/cmake
#sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
#ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
#cmake --version

# Conda environment
#conda create -n soundspaces python=3.8 cmake=3.14.0

# Setup habitat-sim
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
#/bin/bash -c ". activate soundspaces; cd habitat-sim; pip install -r requirements.txt; python setup.py install --headless"
cd habitat-sim
git checkout v0.1.7
pip install -r requirements.txt
python setup.py install

# Install challenge specific habitat-lab
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
#/bin/bash -c ". activate soundspaces; cd habitat-lab; git checkout v0.1.7; pip install -e ."
cd habitat-lab; git checkout v0.1.7; pip install -e .

# Install challenge specific habitat-lab
pwd
git clone --branch master https://github.com/facebookresearch/sound-spaces.git
cd sound-spaces
#git checkout v0.1.2
#/bin/bash -c ". activate soundspaces; cd sound-spaces;pip install -e ."
pip install -e .
# Silence habitat-sim logs
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"
