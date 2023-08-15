version="${1?Specify a docker image version: 'cpu' or 'cuda'}"

image_name="haccel-balkon00"

dataset_dir="/tempdata/balaskas/NN_project/data"
cifar10_chkpt_dir="/tempdata/balaskas/NN_project/cifar10_100_playground/cifar10/models"
cifar100_chkpt_dir="/tempdata/balaskas/NN_project/cifar10_100_playground/cifar100/models"

use_gpus=""
if [ "$version" == "cuda" ]; then
    use_gpus="--gpus all --runtime nvidia"
fi

port_tb_server="4001"
port_tb_container="6006"
port_io_server="4000"
port_io_container="8888"

#docker run --runtime nvidia -it --rm pytorch/pytorch:latest python3
docker run \
	--name haccel-$version \
	-it $use_gpus \
	-v ${dataset_dir}:/workspace/data \
	-v ${cifar10_chkpt_dir}:/workspace/chkpts/cifar10 \
	-v ${cifar100_chkpt_dir}:/workspace/chkpts/cifar100 \
	-p ${port_io_server}:${port_io_container} \
	-p ${port_tb_server}:${port_tb_container} \
	--shm-size 10gb \
	${image_name}:${version} \
	/bin/bash

