#!/bin/bash
git submodule update --init --recursive

LOCAL_DIRECTORY="$HOME/.local"

if [ -d "$LOCAL_DIRECTORY" ]; then
  echo "$LOCAL_DIRECTORY does exist."
fi


#curl -LO https://github.com/bazelbuild/bazel/releases/download/7.7.1/bazel-7.7.1-linux-x86_64
#chmod 777 bazel
#mv bazel-7.7.1-linux-x86_64 bazel
