# Semantic3D

## Data

Data can be downloaded at [http://semantic3d.net](http://semantic3d.net).

In the folder ```semantic3D_utils```:
```
python setup.py install --home="."
```
Then, run the generation script:
```
python prepare_data.py --rootdir /media/liangdao/DATA/Toronto_3D/ConvPoint --savedir /media/liangdao/DATA/Toronto_3D/ConvPoint
```
## Training

The training script is called using:
```
python Toronto_seg.py --rootdir /media/liangdao/DATA/Toronto_3D/ConvPoint/train/pointcloud/ --savedir /media/liangdao/DATA/Toronto_3D/ConvPoint 
```
python Toronto_seg.py --rootdir /media/liangdao/DATA/Toronto_3D/ConvPoint/train/pointcloud/ --savedir /media/liangdao/DATA/Toronto_3D/ConvPoint/SegSmall_8192_nocolorTrue_drop0.5_2022-06-15-23-23-40 --continuetrain

Sematnic3D training, area1 test
```
python semantic3d_seg.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_Domain_8192_nocolorTrue_drop0.5_2022-09-12-19-11-58 --test --savepts
```

### Transfer domain
```
python semantic3d_seg_gan_domain.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/ --finetuning

python semantic3d_seg_gan_domain.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_Domain_8192_nocolorTrue_drop0.5_2022-09-12-19-11-58 --test  --savepts

```
### GAN
python toronto3d_seg_gan.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud/ --savedir /media/liangdao/DATA/Toronto_3D/ConvPoint/

### Fine Tuning
python toronto3d_seg_finetuning.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/Toronto_3D/ConvPoint/SegSmall_8192_nocolorTrue_drop0.5_2022-06-15-23-23-40/SegSmall_8192_finetuning_linearlayer_nocolorTrue_drop0.5_2022-06-23-17-21-20/SegSmall_8192_finetuning_linearlayer_nocolorTrue_drop0.5_2022-06-24-23-51-51 --finetuning


## double discriminator
python toronto3d_seg_2dis.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud/ --savedir /media/liangdao/DATA/Toronto_3D/ConvPoint/



## Test

To predict on the test set (voxelized pointcloud):

```
python semantic3d_seg.py --rootdir /media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud --savedir /media/liangdao/DATA/Toronto_3D/ConvPoint/SegSmall_8192_nocolorTrue_drop0.5_2022-06-15-23-23-40 --test --savepts

```

Finally to generate the prediction files at benchmark format (may take som time): 

```
python semantic3d_benchmark_gen.py --testdir path_to_original_test_data --savedir /path_to_save_dir_benchmark --refdata path_to_data_processed --reflabel path_to_prediction_dir
```






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