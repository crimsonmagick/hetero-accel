#!/bin/bash
set -eou pipefail

env_name="${1:-haccel}"

apt install -y scons libconfig++-dev libboost-dev libboost-iostreams-dev libboost-serialization-dev libyaml-cpp-dev libncurses-dev libtinfo-dev libgpm-dev git build-essential python3-pip
mkdir -p timeloop-accelergy
cd timeloop-accelergy
git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure.git
cd accelergy-timeloop-infrastructure
make pull
cd src/cacti
make
cd ../accelergy
pip3 install --upgrade pip
pip3 install .
cd ../accelergy-aladdin-plug-in/
pip3 install .
cd ../accelergy-cacti-plug-in/
pip3 install .
cp -r ../cacti /opt/conda/envs/${env_name}/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/
cd ../accelergy-table-based-plug-ins/
pip3 install .
cd ../timeloop
cd src/
ln -s ../pat-public/src/pat .
cd ..
scons -j4 --accelergy --static
cp build/timeloop-* /opt/conda/envs/${env_name}/bin
cd ../../..
git clone https://github.com/Accelergy-Project/timeloop-accelergy-exercises.git
accelergy
accelergyTables
python3 -m pip install git+https://github.com/Fibertree-Project/fibertree
pip3 install jupyter
export PATH=$PATH:/opt/conda/envs/${env_name}/bin

