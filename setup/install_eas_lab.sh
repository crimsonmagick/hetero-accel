#!/bin/bash
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

curl -L -o ~/.local/bin/bazel https://github.com/bazelbuild/bazel/releases/download/7.7.1/bazel-7.7.1-linux-x86_64
chmod 777 ~/.local/bin/bazel

#pip3 install pycocotools
