# Semantic3D

## Data

Data can be downloaded at [http://semantic3d.net](http://semantic3d.net).

In the folder ```semantic3D_utils```:
```
python setup.py install --home="."
```
Then, run the generation script:
```
python semantic3d_prepare_data.py --rootdir /media/liangdao/DATA/origin_data/origin --savedir /media/liangdao/DATA/origin_data/convpoint

python semantic3d_prepare_data.py --rootdir /media/liangdao/DATA/small/area123 --savedir /media/liangdao/DATA/small/convpoint

python tummls_prepare_data.py --rootdir /media/liangdao/DATA/small/subarea --savedir /media/liangdao/DATA/small/subarea

```
## Training

The training script is called using:
```
python semantic3d_seg.py --rootdir /media/liangdao/DATA/small/convpoint/train/pointcloud/ --savedir /media/liangdao/DATA/small/convpoint
python semantic3d_seg.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/SegBig_8192_nocolorTrue_drop0.5_2022-08-20-17-52-27 --continuetrain

python semantic3d_seg.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/SegSmall_8192_nocolorTrue_drop0.5_2022-06-06-22-30-46 --continuetrain
```
Sematnic3D training, area1 test
```
python semantic3d_seg.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/SegBig_8192_nocolorTrue_drop0.5_2022-08-20-17-52-27/SegBig_8192_nocolorTrue_drop0.5_2022-08-23-09-18-38 --test  --savepts
```

### Transfer
```
python semantic3d_seg_trans.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/
```
### GAN
python semantic3d_seg_gan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor --finetuning


python semantic3d_seg_gan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_Domain_8192_nocolorTrue_drop0.5_2022-08-01-17-55-41 --continuetrain

python semantic3d_seg_gan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_Domain_8192_nocolorTrue_drop0.5_2022-08-01-17-55-41/SegBig_Domain_8192_nocolorTrue_drop0.5_2022-08-02-16-12-18 --test --savepts

### Fine Tuning
python semantic3d_seg_finetuning.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/SegSmall_8192_nocolorTrue_drop0.5_2022-06-12-23-54-20 --finetuning

python semantic3d_seg_finetuning.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor --finetuning

### Childtuning
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
    python semantic3d_seg_gan_unsup.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-23-21-23-51 --continuetrain
    
    python semantic3d_seg_gan_unsup.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-23-21-23-51/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-08-10-17-56-24/ --test --savepts


## Unsup version domain loss
python semantic3d_seg_gan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor --finetuning

# with PointDAN
    python semantic3d_seg_gan_pointdan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-23-21-23-51 --continuetrain

    python semantic3d_seg_gan_pointdan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-23-21-23-51/SegBig_pointdan_8192_nocolorTrue_drop0.5_2022-08-22-11-02-03 --test --savepts
## Test

To predict on the test set (voxelized pointcloud):

```
python semantic3d_seg.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor --test --savepts


python semantic3d_seg.py --rootdir /media/liangdao/DATA/origin_data/convpoint/test/pointcloud/ --savedir /media/liangdao/DATA/origin_data/convpoint/SegBig_8192_nocolorNone_drop0.5_2022-04-28-02-36-49 --test
```

Finally to generate the prediction files at benchmark format (may take som time): 

```
python semantic3d_benchmark_gen.py --testdir path_to_original_test_data --savedir /path_to_save_dir_benchmark --refdata path_to_data_processed --reflabel path_to_prediction_dir
```


### Very small

python semantic3d_seg_verysmall.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/Prepare/




**note**: the `test_step` parameter is set `0.8`. It is possible to change it. A smaller step of sliding window would produce better segmentation at a the cost of a longer computation time.

## Fusion model

Once models (RGB and without color information) have been trained, it is possible to train a fusion model.

### Training
```
python semantic3d_seg_fusion.py --rootdir path_to_data_processed --savedir path_to_save_dirctory --model_rgb path_to_rgb_model_directory --model_noc path_to_no_color_model_directory
```
### Test
```
python semantic3d_seg_fusion.py --rootdir path_to_data_processeed --savedir path_to_save_dirctory --model_rgb path_to_rgb_model_directory --model_noc path_to_no_color_model_directory --test --savepts
```

## Pretrained models

Pretrained models can be found [here](https://github.com/aboulch/ConvPoint/releases/download/0.1.0/models_SEMANTIC3D_v0.zip).

**Note**: due to change of affiliation and loss of data, these models are given as they are, without any performance guarantee. Particularly, they may not be the ones used in the final version of the paper.