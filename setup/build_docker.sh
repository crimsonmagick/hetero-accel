#!/bin/bash
set -eoux pipefail

version="${1?Specify a docker image version: 'cpu' or 'cuda'}"
private_ssh_key="${2?Specify a private ssh key for GitHub}"

dockerfile="setup/${version}.Dockerfile"
image_tag="haccel-balkon00:$version"

DOCKER_BUILDKIT=1 \
	docker build \
	--file $dockerfile \
	--secret id=ssh_id,src=$private_ssh_key \
	--tag $image_tag \
	.
	#--ssh default=$SSH_AUTH_SOCK \
	#--build-arg SSH_KEY=${private_ssh_key} \

