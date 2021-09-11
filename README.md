# setup-racing-car-ubuntu
this setup support Ubuntu 18.04


## Contents
1. [Install Prerequisites](https://github.com/cauhamau/Racing-Car-2021/blob/set-up/README.md#1-install-prerequisites)
2. [Setup NVIDIA Driver for your GPU](https://github.com/cauhamau/Racing-Car-2021/blob/set-up/README.md#2-setup-nvidia-driver-for-your-gpu)
3. [Install CUDA](https://github.com/cauhamau/Racing-Car-2021/blob/set-up/README.md#3-install-cuda)
4. [Install cuDNN](https://github.com/cauhamau/Racing-Car-2021/blob/set-up/README.md#4-install-cudnn)
5. [Install TensorRT](https://github.com/cauhamau/Racing-Car-2021/blob/set-up/README.md#5-install-tensorrt)
6. [Python and Other Dependencies](https://github.com/cauhamau/Racing-Car-2021/blob/set-up/README.md#6-python-and-other-dependencies)
7. [Deep Learning Frameworks](https://github.com/cauhamau/Racing-Car-2021/blob/set-up/README.md#7-install-deep-learning-frameworks)
    - [PyTorch](https://github.com/cauhamau/Racing-Car-2021/blob/set-up/README.md#pytorch)
    - [TensorFlow](https://github.com/cauhamau/Racing-Car-2021/blob/set-up/README.md#tensorflow)
    - [Darknet for YOLO](https://github.com/cauhamau/Racing-Car-2021/blob/set-up/README.md#darknet-for-yolo)
8. [ROS Melodic](https://github.com/cauhamau/Racing-Car-2021/blob/set-up/README.md#8-ros-melodic)


## 1. Install Prerequisites
Before installing anything, let us first update the information about the packages stored on the computer and upgrade the already installed packages to their latest versions.

    sudo apt-get update
    sudo apt-get upgrade

Next, we will install some basic packages which we might need during the installation process as well in future.

    sudo apt-get install -y build-essential cmake gfortran git pkg-config 

## 2. Setup NVIDIA Driver for your GPU

follow [this video](https://www.youtube.com/watch?v=GljujCLixzE) and [src](./src/Install_Nvidia_Driver.md)
# If Error:
    Ctrl + Alt + F3
    sudo apt-get purge nvidia*
    sudo apt-get autoremove 
    
# Easy set up:
    sudo -i
    ubuntu-drivers autoinstall

## 3. Install CUDA

You can also install CUDA directly from the offline installer, but this is a little easier.

```
sudo apt update
sudo apt upgrade -y

mkdir install ; cd install
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-10-2
```

## 4. Install cuDNN

Download CuDNN here (BOTH the runtime and dev, deb) from: [drive link 1](https://drive.google.com/file/d/1xXtoLcXWTW2EDUaxXmlp1gB9pEXxsWst/view?usp=sharing) [drive link 2](https://drive.google.com/file/d/14nyUfLgjpTWLOH-pthrG33_IMWczZiZM/view?usp=sharing) or [nvidia source](https://developer.nvidia.com/rdp/cudnn-download)

```
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb
```

### Verify installation

This method of installation installs cuda in `/usr/include` and `/usr/lib/cuda/lib64`, hence the file you need to look at is in `/usr/include/cudnn.h`.

```
cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2                                                         
```

## 5. Install TensorRT

Download TensorRT [drive link](https://drive.google.com/file/d/1K1l4esENGP_zTArocVUM4AADvZD35xDW/view?usp=sharing) or [nvidia source](https://developer.nvidia.com/nvidia-tensorrt-7x-download). Use version 7.0.

```
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb
sudo apt update
sudo apt install tensorrt libnvinfer7
```

## Add to .bashrc for TensorRT cuDNN CUDA  - (important)

```
sudo gedit ~/.bashrc # open bashrc and copy lines below
```

```
export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH=$CUDA_HOME/lib64:$DYLD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_RUN_PATH=$CUDA_HOME/lib64:$LD_RUN_PATH
```
save gedit and write command:

```
source ~/.bashrc
```

## 6. Python and Other Dependencies

Install dependencies of deep learning frameworks:

    sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libopencv-dev

Next, we install Python 2 and 3 along with other important packages like boost, lmdb, glog, blas etc.

    sudo apt-get install -y --no-install-recommends libboost-all-dev doxygen
    sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev libblas-dev 
    sudo apt-get install -y libatlas-base-dev libopenblas-dev libgphoto2-dev libeigen3-dev libhdf5-dev 
     
    sudo apt-get install -y python-dev python-pip python-nose python-numpy python-scipy python-wheel python-six
    sudo apt-get install -y python3-dev python3-pip python3-nose python3-numpy python3-scipy python3-wheel python3-six
    
**NOTE: If you want to use Python2, replace the following pip commands with pip2.**

Before we use pip, make sure you have the latest version of pip.

    pip3 install --upgrade pip

Now, we can install all the required python packages for deep learning frameworks:

    pip3 install numpy matplotlib ipython protobuf jupyter mock
    pip3 install scipy scikit-image scikit-learn
    pip3 install opencv-python==4.5.3 opencv-contrib-python==4.5.3
    
Upgrade numpy to the latest version:

    sudo pip3 install --upgrade numpy

## 7. Install Deep Learning Frameworks

### PyTorch  

You can run the commands for installing pip packages `torch` and `torchvision` from [the Quick Start section here](https://pytorch.org/get-started/previous-versions/).

    pip install torch==1.6.0 torchvision==0.7.0

### TensorFlow

#### Quick Install (Not Recommended)

A quick way to install TensorFlow using pip without building is as follows. However this is not recomended as we have several specific versions of GPU libraries to improve performance, which may not be available with the pip builds.

    sudo pip3 install tensorflow-gpu

### Darknet for YOLO

First clone the Darknet git repository.

    git clone https://github.com/pjreddie/darknet.git

Now, to compile Darknet with CUDA, CuDNN and OpenCV support, open the `Makefile` from the `darknet` folder and make the changes as following in the beginning of this file. Also make sure to select the right architecture based on your GPU's compute capibility. For Pascal architecture you may want to use [this version of Darknet by AlexeyAB](https://github.com/AlexeyAB/darknet) and compile with the `CUDNN_HALF=1` flag for 3x speed improvement.

Chage [Makefile](https://github.com/AlexeyAB/darknet/blob/master/Makefile)

    GPU=1 
    CUDNN=1
    OPENCV=1
    LIBSO=1 

Check Nvidia Compute Capability your card and change [lines](https://github.com/AlexeyAB/darknet/blob/master/Makefile#L20)

Once done, just run make from the darknet folder.

    cd darknet
    make

Refer [here](https://pjreddie.com/darknet/yolo/) for more details on running YOLO and training the network.

## 8. ROS Melodic

    wget https://github.com/tonhathuy/setup-racing-car-ubuntu/blob/main/src/setup-ros.sh
    sudo chmod +x ./setup-ros.sh
    ./setup-ros.sh

