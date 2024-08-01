docker run -it --gpus device=0 --cpus=0.000 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 8889:8888 -v /media/robbie/cradle1:/data vxm-docker bash
