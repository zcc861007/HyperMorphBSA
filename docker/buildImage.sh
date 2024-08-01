#!/bin/bash


# This file must be run as sudo in linux

#if [[ "$(docker images -q asa:monai)" != "" ]]; then
#	# do something
#	docker rmi -f asa:monai
#fi

docker build -t vxm-docker .
