version="${1?Specify a docker image version: 'cpu' or 'cuda'}"

dockerfile="setup/${version}.Dockerfile"
image_tag="haccel-balkon00:$version"

docker build --file $dockerfile --tag $image_tag .

