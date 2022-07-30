
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

predict_root= "/media/liangdao/DATA/Toronto_3D/ConvPoint/SegSmall_8192_nocolorTrue_drop0.5_2022-06-15-23-23-40/results/"
original_path = ["/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud/mls2016_8class_20cm_ascii_area1_voxels.npy","/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud/mls2016_8class_20cm_ascii_area2_voxels.npy","/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud/mls2016_8class_20cm_ascii_area3_voxels.npy"]
predict_path= ["mls2016_8class_20cm_ascii_area1_voxels.labels", "mls2016_8class_20cm_ascii_area2_voxels.labels","mls2016_8class_20cm_ascii_area3_voxels.labels"]
for i in range(len(original_path)):
    original_points = np.load(original_path[i])
    original_labels = original_points[:,-1]-1

    values, counts = np.unique(original_labels, return_counts=True)
    print(values,counts)

    File_data = np.loadtxt(predict_root+predict_path[i], dtype=float)

    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(File_data[:,:3])
    labels = File_data.data

    values, counts = np.unique(labels, return_counts=True)
    print(values,counts)
    cm += confusion_matrix(original_labels.ravel(), labels, labels=list(range(N_CLASSES)))

#max_label = 8
#colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
#pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
#o3d.visualization.draw_geometries([pcd])

print(cm)
iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"
print(iou)