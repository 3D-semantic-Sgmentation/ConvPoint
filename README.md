# Semantic Segmentation with Transfer Learning for MLS Point Clouds

## Introduction

This repository propose python scripts for Semantic Segmentation with Transfer Learning for MLS Point Clouds. The library is based on the project [ConvPoint](https://github.com/aboulch/ConvPoint)

## Data

Semantic data can be downloaded at [http://semantic3d.net](http://semantic3d.net).

In the folder ```semantic3D_utils```:
```
python setup.py install --home="."
```
Then, run the generation script:
```
python tummls_prepare_data.py --rootdir /media/liangdao/DATA/Paris_and_Lille --savedir /media/liangdao/DATA/Paris_and_Lille

python semantic3d_prepare_data.py --rootdir /media/liangdao/DATA/small/area123 --savedir /media/liangdao/DATA/small/convpoint

python tummls_prepare_data.py --rootdir /media/liangdao/DATA/small/subarea --savedir /media/liangdao/DATA/small/subarea

```

TUM-MLS can be downloaded at [testdaten](https://www.pf.bgu.tum.de/pub/testdaten.html)


## Training from scratch

The training script is called using:
```
python semantic3d_seg.py --rootdir your_pointcloud_path  --savedir your_save_folder_path
e.g.
python semantic3d_seg.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud  --savedir /media/liangdao/DATA/segmentation/ConvPoint/data
```

## Continue training
Add `--continuetrain` at the end, which means reading a pretrained model from `savedir` and continue update the parameters

```
python semantic3d_seg.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/SegBig_8192_nocolorTrue_drop0.5_2022-08-20-17-52-27 --continuetrain

python semantic3d_seg.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/SegSmall_8192_nocolorTrue_drop0.5_2022-06-06-22-30-46 --continuetrain

```

## Test
Sematnic3D training, area1 test
```
python semantic3d_seg.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SegBig_8192_nocolorTrue_drop0.5_2022-09-14-09-23-17 --test  --savepts
```

### Transfer Learning
```
python semantic3d_seg_trans.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/
```

### GAN-based
python semantic3d_seg_gan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor --finetuning


python semantic3d_seg_gan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_Domain_8192_nocolorTrue_drop0.5_2022-08-01-17-55-41 --continuetrain

python semantic3d_seg_gan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_Domain_8192_nocolorTrue_drop0.5_2022-08-01-17-55-41/SegBig_Domain_8192_nocolorTrue_drop0.5_2022-08-02-16-12-18 --test --savepts

### Fine Tuning
python semantic3d_seg_finetuning.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_8192_finetuning_linearlayer_nocolorTrue_drop0.5_2022-08-30-17-17-00 --finetuning

python semantic3d_seg_finetuning.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor --finetuning


### Childtuning
python semantic3d_seg_childtuning.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/ --finetuning

python semantic3d_seg_childtuning.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/ --finetuning

### MMD
python semantic3d_seg_mmd.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor --finetuning


## Reverse
update discriminator when semantic3D training; load pretrained model
python semantic3d_seg_gan_reverse.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/ --finetuning


## Unsup version unspu Gan
### first step:
    python semantic3d_seg_gan_unsup_step1.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-23-21-23-51 --continuetrain
### second Step:
    python semantic3d_seg_gan_unsup.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor --finetuning
    python semantic3d_seg_gan_unsup.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-09-30-20-00-25 --continuetrain
    
    python semantic3d_seg_gan_unsup.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-23-21-23-51/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-08-25-18-44-10 --test --savepts

'''

<!-- test_step=0.5 8829+7074+4521  Time: 9706.95879817009
test_step=0.8 3674+2955+1906 Time: 5581.792199134827/4070.75
test_step=1.0 2421+1951+1263 2698.98
#2297+1847+1191 Time: 3747.276200532913/
test_step=1.2 1724+1391+909 3533.3705315589905/ 2629.451016187668 -->

## Unsup version domain loss
python semantic3d_seg_gan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor --finetuning

# with PointDAN
    python semantic3d_seg_gan_pointdan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-23-21-23-51 --continuetrain

    python semantic3d_seg_gan_pointdan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-09-25-18-33-05 --test --savepts

## Test
To predict on the test set (voxelized pointcloud):
```
python semantic3d_seg.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_8192_childtuning_nocolorTrue_drop0.5_2022-08-30-17-20-24 --test --savepts

python semantic3d_seg.py --rootdir /media/liangdao/DATA/origin_data/convpoint/test/pointcloud/ --savedir /media/liangdao/DATA/origin_data/convpoint/SegBig_8192_nocolorNone_drop0.5_2022-04-28-02-36-49 --test
```

Finally to generate the prediction files at benchmark format (may take som time): 
```
python semantic3d_benchmark_gen.py --testdir path_to_original_test_data --savedir /path_to_save_dir_benchmark --refdata path_to_data_processed --reflabel path_to_prediction_dir
```


**note**: the `test_step` parameter is set `0.8`. It is possible to change it. A smaller step of sliding window would produce better segmentation at a the cost of a longer computation time.

## Pretrained models

Pretrained models can be found [here](https://github.com/aboulch/ConvPoint/releases/download/0.1.0/models_SEMANTIC3D_v0.zip).

**Note**: due to change of affiliation and loss of data, these models are given as they are, without any performance guarantee. Particularly, they may not be the ones used in the final version of the paper.