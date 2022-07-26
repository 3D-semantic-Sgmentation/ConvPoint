
import sys
sys.path.append('../../')

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import metrics as metrics

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
