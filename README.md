# AccelerateOED
code for paper "A data-driven approach to accelerating Optimal Experimental Design for Uncertain Kuramoto model"

## Environment
Python 3.8.8  
Ubuntu 18.04.1  
GPU GeForce RTX 2080 Ti; CUDA Version: 10.2  

### Install Dependencies

create conda enviroment
```bash
conda create -n MOCU python=3.8
conda activate MOCU
```

Install PyCUDA (https://medium.com/leadkaro/setting-up-pycuda-on-ubuntu-18-04-for-gpu-programming-with-python-830e03fc4b81)  
for parallel computing
```bash
# preparation 
# install cuda toolkit
sudo apt install nvidia-cuda-toolkit 
# install gcc compiler
sudo apt-get install build-essential binutils gdb
# install nvcc compiler
sudo apt-get install freeglut3 freeglut3-dev libxi-dev libxmu-dev
pip install six

sudo apt-get install build-essential python-dev python-setuptools libboost-python-dev libboost-thread-dev -y
pip install pycuda

pip install pandas
 ```
It's better to test PyCUDA before the following steps.  
Now you are able to run sampling-RK,the original method to compute MOCU.

Install PyTorch and PyTorch Geometric(https://pytorch.org/  https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)  
for training and testing message passing model

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
# I went through a numpy version comflict here, since pip installed numpy=1.20.1 while installing pycuda, 
# and conda installed numpy=1.19.1 while installing pytorch.
# Solved by pip uninstall numpy and conda install numpy=1.20.1.
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html 
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html   
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html  
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-geometric
```


 
