#!/bin/bash

haccel_dir="/workspace/hetero-accel"

jupyter lab \
	--ip 0.0.0.0 \
	--port 8888 \
	--allow-root \
	--notebook-dir $haccel_dir \
	--preferred-dir $haccel_dir
#jupyter lab \
#	--ip 0.0.0.0 \
#	--port 8888 \
#	--allow-root \
#	--notebook-dir $haccel_dir/accelergy-timeloop-infrastructure/timeloop-accelergy-exercises/workspace \
#	--preferred-dir $haccel_dir/accelergy-timeloop-infrastructure/timeloop-accelergy-exercises/workspace/exercises/2020.ispass \
