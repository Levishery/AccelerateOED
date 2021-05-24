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
pip  install openpyxl
 ```
It's better to test PyCUDA before the following steps.  
Now you are able to run sampling-RK,the original method to compute MOCU.

Install PyTorch and PyTorch Geometric(https://pytorch.org/  https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)  
for training and testing message passing model

First check your CUDA version with
```bash
nvcc --version
```
If it's 10.2, please use the following command without change. Otherwise please change "10.2" or "cu102" to your CUDA version(like "10.1" or "cu101").
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


### Creat Dataset
use make_data.py in the forlder for N=5 and N=7
See make_data.py for data distribution.
Use data/Graph_dataset.py to combine and convert the data to class torch_geometric.data, and seperate training and test sets.

### Training
We used a two-stage training, where the first stage use dataset contains only 5-oscillator system, while the second stage use both. The target MOCU values arenormalized to mean 0 and variance 1.We use the Adam optimizer with learning rate 0.001 and batch size 128 intraining for 400 epoch. The weight for ranking constrain loss is set to 0.0001.
```bash
cd model
python MP_train.py  --name cons5 --data_path ../Dataset/70000_5o_train.pth --EPOCH 400 --Constrain_weight 0.0001
python MP_train.py  --pretrain cons5 --name consmixed --data_path ../Dataset/56000_mixed.pth --EPOCH 400 --Constrain_weight 0.0001
```
The models and loss curves will be saved in Experiment/name

### Test
```bash
# MSE on testset
python MP_train.py  --pretrain consmixed --test_only --data_path ../Dataset/2000_7o_test.pth
# Ability to preserve Ranking. Remember to change the model path and data path.
python vali_model.py
# OED simulation. Remember to change the model path in findMPSequence.py.
mkdir results
cd N5ForShare/N7ForShare
python runMainForPerformanceMeasure.py
# The mean mocu matrrix will be saved in results/mean_MOCU.txt
```
In the OED simulation experiment, I created a child process to run the neural network. Because the cuda context initialed by pycuda will be destroyed if pytorch initial it again, the child process force them to initilize their own CUDA context. There might be a more elegent solution.
