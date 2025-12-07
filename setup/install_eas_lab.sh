#!/bin/bash
git submodule update --init --recursive

create_directory () {
  pending_dir=$0
  if [ -d "$pending_dir" ]; then
    echo "$pending_dir does exist."
    mkdir "$pending_dir"
    echo "$pending_dir created."
  else
    echo "$pending_dir exists"
  fi
}

local_directory="$HOME/.local"

create_directory "$local_directory"

LOCAL_DIRECTORY="$local_directory/bin"

create_directory "$local_directory"

curl -LO -o ~/.local/bin/bazel https://github.com/bazelbuild/bazel/releases/download/7.7.1/bazel-7.7.1-linux-x86_64
chmod 777 ~/.local/bin/bazel

pip3 install pycocotools
