from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import open3d as o3d
import random
import os
import torch



def draw_features(features, labels, title = "0"):

    tsne = TSNE(n_components=2).fit_transform(features)
    colors_per_class = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))
        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range

        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    fig = plt.figure()
    ax = fig.add_subplot(111)
        
    # for every class, we'll add a scatter plot separately
    for label in labels:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only

        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
    
        # convert the class color to matplotlib format
        #color = np.array(colors_per_class[label], dtype=np.float) 
    
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, label=label)
    
    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plt.title(title)

    plt.show()


def showSample(folder, filelist, bs, npoints=8192):

    index = random.randint(0, len(filelist)-1)
    pts = np.load(os.path.join(folder, filelist[index]))
    print(pts.shape)
    # get the labels
    lbs = pts[:, 6].astype(int) # the generation script label starts at 1

    # get the point coordinates
    pts = pts[:, :3]


    # pick a random point

    pt_id = random.randint(0, pts.shape[0]-1)
    pt = pts[pt_id]

    # create the mask
    mask_x = np.logical_and(pts[:,0]<pt[0]+bs/2, pts[:,0]>pt[0]-bs/2)
    mask_y = np.logical_and(pts[:,1]<pt[1]+bs/2, pts[:,1]>pt[1]-bs/2)
    mask = np.logical_and(mask_x, mask_y)
    pts = pts[mask]
    lbs = lbs[mask]
    print(pts.shape)
    # random selection
    choice = np.random.choice(pts.shape[0], npoints, replace=True)
    pts = pts[choice]
    lbs = lbs[choice]

    pts = torch.from_numpy(pts).float()
    lbs = torch.from_numpy(lbs).long()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    max_label = 8
    colors = plt.get_cmap("tab20")(lbs / (max_label if max_label > 0 else 1))
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

# original_path = "/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud"
# #original_files = ["mls2016_8class_20cm_ascii_area1_voxels.npy","mls2016_8class_20cm_ascii_area1_2_voxels.npy","mls2016_8class_20cm_ascii_area2_voxels.npy","mls2016_8class_20cm_ascii_area3_voxels.npy"]
# train_files = ["bildstein_station1_xyz_intensity_rgb_voxels.npy","bildstein_station3_xyz_intensity_rgb_voxels.npy","bildstein_station5_xyz_intensity_rgb_voxels.npy","domfountain_station1_xyz_intensity_rgb_voxels.npy","domfountain_station2_xyz_intensity_rgb_voxels.npy","domfountain_station3_xyz_intensity_rgb_voxels.npy","neugasse_station1_xyz_intensity_rgb_voxels.npy","sg27_station1_intensity_rgb_voxels.npy","sg27_station2_intensity_rgb_voxels.npy","sg27_station4_intensity_rgb_voxels.npy",
# "sg27_station5_intensity_rgb_voxels.npy","sg27_station9_intensity_rgb_voxels.npy","sg28_station4_intensity_rgb_voxels.npy","untermaederbrunnen_station1_xyz_intensity_rgb_voxels.npy","untermaederbrunnen_station3_xyz_intensity_rgb_voxels.npy"]
# showSample(original_path, train_files, 32, 8192)