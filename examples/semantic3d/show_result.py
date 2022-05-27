import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


File_data = np.loadtxt("/media/liangdao/DATA/small/convpoint/area23_train_area1_test/results/mls2016_8class_20cm_ascii_area1_voxels.npy_pts.txt", dtype=float)
#File_data = np.loadtxt("/media/liangdao/DATA/small/convpoint/train/pointcloud_txt/mls2016_8class_20cm_ascii_area1_voxels.txt", dtype=float)


print(File_data)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(File_data[:,0:3])
labels = File_data[:, 3]+1
max_label = 8
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])