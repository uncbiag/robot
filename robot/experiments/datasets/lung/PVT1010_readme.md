# PVT1010
The PVT1010 dataset includes 1,010 pairs of high resolution inhale/exhale lung vascular trees extracted from 3D computed tomography (CT) images.

Details on how we extracted the lung vascular trees as high-resolution 3D point clouds from the raw CT images can be found at [Suppl.A.1.](https://arxiv.org/pdf/2111.00648.pdf#subsection.A.1)

## Download dirlab COPD data
We use public [DirLab-COPD Gene dataset](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/copdgene.html) to evaluate our registration results. Please follow the official instruction to download the data.

DirLab-COPD includes 10 pair cases. For each pair, 300 expert annotated landmarks that are in correspondence with each other. 

These 10 cases are used as test cases.

## Download PVT1010
The full dataset can be accessed at
```
https://drive.google.com/drive/folders/1iA6VSAjmMP_iEHp1iae4anu-RELxtgTq?usp=sharing
```
The vascular tree reconstructions that are used in this study were part of the COPDGene study (NCT00608764). This study has been IRB approved and participants have provided their consent.

 
## Prepare data 

```
cd robot/experiments/datasets/lung
sh prepare_lung_data.sh DIRLAB_DATA_FOLDER PVT1010_DATA_FOLDER DATASPLITS_FOLDER
```
* ```DIRLAB_DATA_FOLDER``` refers to unzipped dirlab data, including 10 data folders: copd1-copd10.
* ```PVT1010_DATA_FOLDER``` refers to PVT1010 data, including 2020 vtk (1010 pairs) data files.
* ```DATASPLITS_FOLDER``` (output folder) records the data splits. Four folders will be created in ```DATASPLITS_FOLDER```, which refers to the "train", "val", "test", "debug"* splits.

*Here the "debug" split refers to a subset of training set, to help diagnose the model behavior on training set.


# Visualization and analysis
1. lung_local_plot.py visualizes the anistropic kernel on vessels
2. lung_data_anlsysis.py computes and matches the lung radius distribution