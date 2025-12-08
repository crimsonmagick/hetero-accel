#!/bin/bash

VENV_NAME=".venv"

git submodule update --init --recursive

create_directory () {
  pending_dir=$1
  if [ ! -d "$pending_dir" ]; then
    echo "$pending_dir does not exist."
    mkdir "$pending_dir"
    echo "$pending_dir created."
  else
    echo "$pending_dir exists"
  fi
}

local_directory="$HOME/.local"
create_directory "$local_directory"
create_directory "$local_directory/bin"
create_directory "$local_directory/share"
mkdir -p "$HOME/.local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/"

curl -L -o "$HOME/.local/bin/bazel" "https://github.com/bazelbuild/bazel/releases/download/7.7.1/bazel-7.7.1-linux-x86_64"
chmod 777 "$HOME/.local/bin/bazel"

# create venv, install project requirements
if [ ! -d "$VENV_NAME" ]; then
    echo "$VENV_NAME venv does not exist."
    python3 -m venv .venv
    if [ $? -eq 0 ]; then
      echo "Created $VENV_NAME successfully"
    else
      echo "Failed to create venv $VENV_NAME. Ending script early..."
      exit
    fi
else
  echo "$VENV_NAME venv already exists, continuing."
fi

. $VENV_NAME/bin/activate
pip3 install -r setup/requirements.txt

make -C accelergy-timeloop-infrastructure/src/cacti -f accelergy-timeloop-infrastructure/src/cacti/makefile
pip3 install accelergy-timeloop-infrastructure/src/accelergy
pip3 install accelergy-timeloop-infrastructure/src/accelergy-aladdin-plug-in
pip3 install accelergy-timeloop-infrastructure/src/accelergy-cacti-plug-in
cp -r accelergy-timeloop-infrastructure/src/cacti "$HOME/.local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/"
pip3 install accelergy-timeloop-infrastructure/src/accelergy-table-based-plug-ins
ln -s  accelergy-timeloop-infrastructure/src/timeloop/pat-public/src/pat accelergy-timeloop-infrastructure/src/timeloop/src
scons -C accelergy-timeloop-infrastructure/src/timeloop -j4 --accelergy --static
cp -r accelergy-timeloop-infrastructure/src/timeloop-* ~/.local/bin

cd generalizedassignmentsolver
bazel build -- ...
cd ..

deactivate