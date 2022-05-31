# Semantic3D

## Environment

### Platform

The code was tested on Ubuntu 16.04 with Anaconda.

### Dependencies

- Pytorch
- Scikit-learn for confusion matrix computation, and efficient neighbors search  
- TQDM for progress bars
- PlyFile
- H5py

All these dependencies can be install via conda in an Anaconda environment or via pip.

### The library

#### Nearest neighbor module

The ```nearest_neighbors``` directory contains a very small wrapper for [NanoFLANN](https://github.com/jlblancoc/nanoflann) with OpenMP.
To compile the module:
```
cd nearest_neighbors
python setup.py install --home="."
```

In the case, you do not want to use this C++/Python wrapper. You still can use the previous version of the nearest neighbors computation with Scikit Learn and Multiprocessing, python only version (slower). To do so, add the following lines at the start of your main script (e.g. ```modelnet_classif.py```):
```
from global_tags import GlobalTags
GlobalTags.legacy_layer_base(True)
```
#### Semantic3D utils
In the folder ```semantic3D_utils```:
```
python setup.py install --home="."

```

## Data

Data is in the folder ../smalldata.


## Training
First of all, cd to examples/semantic3d folder.
Then the training script is called using:
```
python semantic3d_seg.py --rootdir ../../smalldata/train/pointcloud --savedir ../../smalldata/
```
