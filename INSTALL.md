
## Installation

We provide three ways to install robot: 1) a partial installation 2) a custom installation  and 3) using Docker.


### Partial installation
If you only work on registration tasks, we can use following steps to install RobOT.

* Important. Before the installation, make sure the cuda-toolkit is installed. You can check if it is installed via "nvcc --version" in the terminal (see step 0 if nvcc is not yet installed). The cuda compiler version it shows may be different from your cuda driver version shown at "nvidia-smi". Please make sure that torch, and torch_scatter are installed under the same cuda version as the one of nvcc. (Note that if your nvcc version is 11.2 as pytorch and torch_scatter of version 11.2 are not released, you can install any available version compiled with cuda 11.*)
  
We assume all the following installation is under a conda virtual environment, e.g.
```
conda create -n robot python=3.6
conda activate robot
```
1. (Optional) if you cannot find nvcc in the system, you can install it via
```angular2html
conda install -c conda-forge cudatoolkit-dev=11.2
```

2. Now, we can install robot with the following commands
```
git clone https://github.com/uncbiag/robot.git
cd robot/robot
pip install -r requirement.txt
cd ..
cd pointnet2/lib
python setup.py install
```
*if you use Fedora 33, you may meet a bug caused by a specific gcc version, you may need to downgrade the gcc version via *dnf downgrade gcc*

3. Install Keops [link](https://www.kernel-operations.io/keops/python/installation.html)
   After the installation please run the following test to make sure Keops work):
```
import pykeops
pykeops.test_torch_bindings()  
```

4. torch-scatter needs to be installed, see [here](https://github.com/rusty1s/pytorch_scatter).
E.g. for cuda 11.0, 
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
```

### Custom installation
A full installation involves pytorch3D (for general point cloud deep learning tasks).

* Important. Before the installation, make sure the cuda-toolkit is installed. You can check if it is installed via "nvcc --version" in the terminal (see step 0 if nvcc is not yet installed). The cuda compiler version it shows may be different from your cuda driver version shown at "nvidia-smi". Please make sure that torch, pytorch3d, keops, and torch_scatter are installed under the same cuda version as the one of nvcc. (Note that if your nvcc version is 11.2 as pytorch and torch_scatter of version 11.2 are not released, you can install any available version compiled with cuda 11.*)
  
We assume all the following installation is under a conda virtual environment, e.g.
```
conda create -n robot python=3.6
conda activate robot
```
1. (Optional) if you cannot find nvcc in the system, you can install it via
```angular2html
conda install -c conda-forge cudatoolkit-dev=11.2
```

2. For general prediction tasks, pytorch3d needs to be installed first [link](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md). 
   Please install all necessary packages mentioned there. Essentially, pytorch3d needs pytorch to be installed first; we test using pytorch version 1.7.1. Make sure pytorch is compiled with the correct cuda version, e.g. with nvcc version=11.1. We can install pytorch
    via *conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0*. However, if you have already installed cudatoolkit-dev=11.2 then don't include cudatoolkit=11.0 for the pytorch installation.

3. Now, we can install robot with the following commands
```
git clone https://github.com/uncbiag/robot.git
cd robot/robot
pip install -r requirement.txt
cd ..
cd pointnet2/lib
python setup.py install
```
*if you use Fedora 33, you may meet a bug caused by a specific gcc version, you may need to downgrade the gcc version via *dnf downgrade gcc*

4. Install Keops [link](https://www.kernel-operations.io/keops/python/installation.html)
   After the installation please run the following test to make sure Keops work):
```
import pykeops
pykeops.test_torch_bindings()  
 
```

5. torch-scatter needs to be installed, see [here](https://github.com/rusty1s/pytorch_scatter).
E.g. for cuda 11.0, 
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
```

### Docker (doesn't support open3d and teaser++)

If you are familiar with docker, it will likely be much easier to run robot in docker.

1. Push the lastest robot image from dockerhub
```
docker push hbgtjxzbbx/robot:v0.5
```
2. Run docker locally
```
docker run --privileged --gpus all -it --rm  -v /home/zyshen/proj/robot:/proj/robot -v /home/zyshen/data/lung_data:/data/lung_data hbgtjxzbbx/robot:v0.5
```
* Here -v refers to the map between the local path and the docker path.
  We map a code path and a data path based on my local env. Please modify the local path based on your own environment.

3. Compile CUDA code (if you use Fedora 33, you may meet a bug from a specific gcc version, you may need to downgrade gcc version via *dnf downgrade gcc*)
```
cd pointnet2/lib
python setup.py install
```

### Optional third party packages
For full function support, additional packages need to be installed

1. Install [probreg](https://github.com/neka-nat/probreg)
   
   (the open3d version in probreg is old, some APIs have been deprecated, we recommend to install from source and fix open3d minor crashes manually)

2. Install Teaser++ [link](https://teaser.readthedocs.io/en/master/installation.html)
3. Install Open3d [link](http://www.open3d.org/docs/0.7.0/getting_started.html)
   
