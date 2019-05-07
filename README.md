# amcl

## Introduction
This is a CUDA implementation of ROS `amcl` package.  
`amcl` is a probabilistic localization system for a robot moving in 2D. It implements the adaptive (or KLD-sampling) Monte Carlo localization approach (as described by Dieter Fox), which uses a particle filter to track the pose of a robot against a known map.

## Requirements
* CUDA ToolKit

## Preparation
Please specify appropriate `arch` and `code` for your GPU in `CMakeLists.txt`.  
You can find information in <https://developer.nvidia.com/cuda-gpus>.  
An example for GeForce GTX 1060(`Compute Capability` is 6.1) is shown below.

```cmake
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61")
```

## Build Instructions
```shell
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/src
$ git clone https://github.com/atinfinity/amcl.git
$ cd ~/catkin_ws
$ catkin_make
$ source devel/setup.bash
```
