## RobOT

This is the  repository for the paper "Accurate Point Cloud Registration with Robust Optimal Transport".

The repository provides a general framework for point cloud/mesh registration, supporting both optimization- and learning-based registration approaches.  It also provides a general framework for deep prediction tasks, e.g. for 3D landmark detection. 

Besides, the repository to https://github.com/uncbiag/shapmagn contains our latest development code.

### PVT1010 Dataset
We release a large lung vessel tree dataset PVT1010. For usage, please refer to [PVT1010_readme.md](robot/experiments/datasets/lung/PVT1010_readme.md).

The PVT1010 dataset includes 1,010 pairs of high resolution inhale/exhale lung vascular trees extracted from 3D computed tomography (CT) images.

10 cases corresponding to the public DirLab-COPD Gene dataset which includes, for each pair, 300 expert annotated landmarks that are in correspondence with each other. These 10 cases are used as test cases.


### Supported approaches:

#### Optimization-based approaches:
1. Spline registration (supports anisotropic kernel)
2. Fluid registration (supports LDDMM)
3. Coherent Point Drift
4. Robust Optimal Transport (RobOT)
5. RobOT Projection (rigid, spline, LDDMM)
6. RobOT-based uniform mass sampling
7. Third party approaches (Probreg, Open3d, Teaser++)


#### Deep learning-based registration approaches:
1. Spline registration (supports anisotropic kernel)
2. Fluid registration (supports LDDMM)
3. Point cloud feature learning (self-supervised)
4. End-to-end registration (prealign, spline/LDDMM, postprocessing)

#### Other deep learning-based approaches:
1. Landmark Detection


## Installation


For detailed instructions refer to [INSTALL.md](INSTALL.md).

Set `ROBOT_PATH` in [global_variable.py](robot/global_variable) to local robot path.

## Demo
We provide a series of demos, which can be found in the directory robot/demos
Here are two examples demonstrating how to run the optimization-based demos :

### Optimization demos
```
cd robot/robot/demos
python 2d_toy_reg.py
python ot_fun.py
python toy_reg.py
python partial_prealign_reg.py
python lung_gradient_flow.py
python lung_reg.py
python flyingkitti_reg.py
python ot_sampling.py
```

### Deep demo on Lung vessel tree

Here is a <em>toy<em> example on training deep feature learning on 10 dirlab pairs:
```
python run_task.py -ds ./demos/data/lung_data/lung_dataset_splits -o ./demos/output/training_feature_learning_on_one_case -tn deepfeature_pointconv_train -ts ./demos/settings/lung/deep_feature_training -g 0
```

Here is an example on robust optimal transport based deep feature projection (spline) on 10 dirlab pairs:
```
python run_eval.py -dj ./demos/data/lung_data/lung_dataset_splits/test/pair_data.json -o ./demos/output/test_feature_projection_one_case/deepfeature_pointconv_projection -ts ./demos/settings/lung/deep_feature_projection -g 0
```

Here is a <em>toy<em> example on training a pretrained deep LDDMM flow network on 10 dirlab pairs:

```
python run_task.py  -ds ./demos/data/lung_data/lung_dataset_splits -o ./demos/output/train_deepflow_on_one_case -tn deepflow_pwc_lddmm -ts ./demos/settings/lung/deep_lddmm_flow   -g 0
```

Here is a example on evaluating a deep LDDMM flow network on 10 dirlab pairs:

```
python run_eval.py -dj ./demos/data/lung_data/lung_dataset_splits/test/pair_data.json -o ./demos/output/test_deepflow_on_one_case/deepflow_pwc_lddmm -ts ./demos/settings/lung/deep_lddmm_flow  -m   ./demos/pretrained_models/lung_model/pretrained_deep_lddmm -g 0
```

### Deep demo on Kitti

#### Preparing the Kitti data
We need to prepare the data of KITTI Scene Flow 2015 first. Here we follow the instructions in [PointPWC](https://github.com/DylanWusee/PointPWC).

Download and unzip [KITTI Scene Flow Evaluation 2015](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) to directory `RAW_DATA_PATH`.
Run the following script for KITTI data preprocessing; the processed data will be saved in `PROCESSED_DATA_PATH`; a robot compatible format is saved in `SHAPMAN_INTPUT_PATH`:

```
cd robot/experiments/datasets/flying3d_and_kitti/flyingkitti_nonocc
python process_kitti_raw_data.py RAW_DATA_PATH PROCESSED_DATA_PATH
python prepare_data_test_on_kitti.py RAW_DATA_PATH PROCESSED_DATA_PATH SHAPMAN_INTPUT_PATH
```

Here is an example on evaluating a spline flow network on 142 Kitti pairs:

```
python run_eval.py -dj SHAPMAN_INTPUT_PATH/pair_data.json -o ./demos/output/test_deepflow_on_kitti/deep_spline -ts ./demos/settings/kitti/deep_spline_flow  -m ./demos/pretrained_models/kitti_model/pretrained_deep_spline -g 0
```

Here is an example on evaluating a displacement network (PWC) that combines OT prealignement and OT postprocessing on 142 Kitti pairs:
```
python run_eval.py -dj SHAPMAN_INTPUT_PATH/pair_data.json -o ./demos/output/test_deepflow_on_kitti/deep_pwc -ts ./demos/settings/kitti/official_released_pwc_model_with_ot_prealigned_and_post  -m ./demos/pretrained_models/kitti_model/PointConv_726_0.0463.pth -g 0
```

## Gallery

Here are some examples on supported functions:

**[Robust Optimal Transport (RobOT)](robot/demos/ot_fun.py)**:

 <img src=".github/shape_transfer.gif" alt="shape_transfer"  width="500" height="400"/>

**[RobOT Projection (Partial Rigid Registration)](robot/demos/partial_prealign_reg.py)**:

 ![node](.github/partial_bunny.gif) 

**[RobOT Projection (Spline, LDDMM)](robot/demos/toy_reg.py)**:

 ![node](.github/ot_projection.png) 

**Lung vessel Registration (60,000 points)**:

 ![node](.github/lung_vessel_reg.gif) 
 
**[Scene Flow Estimation](robot/demos/flyingkitti_reg.py)**:

  <img src=".github/scene_flow.gif" alt="scene_flow"  width="700" height="300"/> 


**Self-supervised Feature Learning (60,000 points)**:

 <img src=".github/lung_fea.gif" alt="lung_fea"  width="500" height="300"/> 
 
**[Synthesized Data Pair Generation](robot/experiments/datasets/lung/lung_data_aug.py)**:
 
 ![node](.github/synthesize_data.png) 

**Facial Landmark Detection (160,000 points)**:

 <img src=".github/face_landmark.gif" alt="face_landmark"  width="300" height="300"/>
