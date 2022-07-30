
import sys
sys.path.append('../../')

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import metrics as metrics


def show_image():
    #File_data = np.loadtxt("/media/liangdao/DATA/small/convpoint/train/pointcloud_txt/mls2016_8class_20cm_ascii_area1_voxels.txt", dtype=float)
    N_CLASSES=8
    cm = np.zeros((N_CLASSES, N_CLASSES))

    original_path = "/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud/"
    original_files = ["mls2016_8class_20cm_ascii_area2_voxels.npy"]

    predict_root= "/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_8192_finetuning_linearlayer_nocolorTrue_drop0.5_2022-07-04-10-32-07/results_tum/"
    predict_file= ["mls2016_8class_20cm_ascii_area2_voxels.npy_pts.txt"]
    for i in range(len(predict_file)):
        #point = np.load(predict_root + predict_filee[i])
        point = np.loadtxt(predict_root+ predict_file[i])
        print(point)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point[:,:3])
        labels =point[:,-1]


        max_label = 8
        colors = plt.get_cmap("tab20")(labels/(max_label if max_label > 0 else 1))
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd])


    # print(cm)
    # iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"
    # print(iou)
    # print(metrics.stats_iou_per_class(cm)[1])


def statistic_iou():
    N_CLASSES=8
    cm = np.zeros((N_CLASSES, N_CLASSES))

    original_path = "/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud/"
    original_files = ["mls2016_8class_20cm_ascii_area1_voxels.npy","mls2016_8class_20cm_ascii_area2_voxels.npy","mls2016_8class_20cm_ascii_area3_voxels.npy"]
    # original_files = [ "bildstein_station1_xyz_intensity_rgb_voxels.npy",
    #         "bildstein_station3_xyz_intensity_rgb_voxels.npy",
    #         "domfountain_station1_xyz_intensity_rgb_voxels.npy",
    #         "domfountain_station3_xyz_intensity_rgb_voxels.npy",
    #         "neugasse_station1_xyz_intensity_rgb_voxels.npy",
    #         "sg27_station1_intensity_rgb_voxels.npy",
    #         "sg27_station5_intensity_rgb_voxels.npy",
    #         "untermaederbrunnen_station1_xyz_intensity_rgb_voxels.npy",
    #         "bildstein_station5_xyz_intensity_rgb_voxels.npy",
    #         "domfountain_station2_xyz_intensity_rgb_voxels.npy",
    #         "sg27_station4_intensity_rgb_voxels.npy",
    #         "sg27_station2_intensity_rgb_voxels.npy",
    #         "sg27_station9_intensity_rgb_voxels.npy",
    #         "sg28_station4_intensity_rgb_voxels.npy",
    #         "untermaederbrunnen_station3_xyz_intensity_rgb_voxels.npy",]
    predict_root= "/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor"+"/results/"

    print(predict_root)
    # predict_file= ["mls2016_8class_20cm_ascii_area1_2_voxels.labels","mls2016_8class_20cm_ascii_area2_voxels.labels","mls2016_8class_20cm_ascii_area3_voxels.labels"]

    for i in range(len(original_files)):
        original_points = np.load(original_path+original_files[i])
        original_labels = original_points[:,-1]-1

        values, counts = np.unique(original_labels, return_counts=True)
        print(values,counts)

        File_data = np.loadtxt(predict_root+original_files[i].replace(".npy",".labels"), dtype=float)

        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(File_data[:,:3])
        labels = File_data.data

        values, counts = np.unique(labels, return_counts=True)
        print(values,counts)
        cm += confusion_matrix(original_labels.ravel(), labels, labels=list(range(N_CLASSES)))


    print(cm)
    iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"
    print(iou)
    print(metrics.stats_iou_per_class(cm)[1])

statistic_iou()