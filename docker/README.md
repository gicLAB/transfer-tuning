#### Docker 

Create the Docker image with:

``` sh
docker build -t transfer-tuning/tvm:latest -f docker/Dockerfile.main_cpu . 
```

Run the shell for the container in the directory above this:

``` sh
sudo docker run \
    -v ${PWD}/tvm-transfer-tuning:/tvm \
    -v ${PWD}/transfer-tuning/:/workspace \
    -p 8888:8888 -p 9190:9190 -ti transfer-tuning/tvm:latest bash -i        
```


``` sh
sudo docker run \
    --runtime=nvidia \
    --ipc=host \
    -v ${PWD}/tvm-transfer-tuning:/tvm \
    -e NVIDIA_VISIBLE_DEVICES=1 \
    -v ${PWD}/transfer-tuning/:/workspace \
     -p 8888:8888 -p 9190:9190 -ti transfer-tuning/tvm-gpu:latest bash -i
```

### Run the test

``` sh
python3 /workspace/src/exploration/gemm_autoschedule.py

```
