
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import sys
sys.path.append('../../')
from utils import metrics as metrics
# 2686791
# 2687291
folder = "/media/liangdao/DATA/small/convpoint/SegSmall_8192_nocolorNone_drop0.5_2022-05-01-22-46-20/results"
original_folder = "/media/liangdao/DATA/small/convpoint/train/pointcloud"
filelist_test = [
        # "area2_voxels.npy",
        # "area3_voxels.npy",
        "mls2016_8class_20cm_ascii_area3_voxels.labels",
        "mls2016_8class_20cm_ascii_area3_voxels.npy",
        "mls2016_8class_20cm_ascii_area3_voxels.npy_pts.txt"

        ]
        
pre_labels_path = os.path.join(folder, filelist_test[0])

a_file = open(pre_labels_path, "r")

pre_pts_path = os.path.join(folder, filelist_test[2])

pts_file = open(pre_pts_path, "r")
pred_pts=[]
for line in pts_file:
    stripped_line = line.strip()
    pred_pts.append(stripped_line)

pred_labels = []

labels_path = os.path.join(original_folder, filelist_test[1])
pts_true = np.load(labels_path)
#print(pts_true)
result = np.load(labels_path)[:,6].astype(int)-1

correct = 0

for line in a_file:
    stripped_line = line.strip()
    pred_labels.append(int(stripped_line))

for i in range(len(result)):
    #print(pred_pts[i], "True:", pts_true[i])
    if pred_labels[i] == result[i]:
        correct+=1

print(correct/len(result))
cm = confusion_matrix(result, pred_labels, labels=list(range(8)))
print(cm)

oa = f"{metrics.stats_overall_accuracy(cm):.5f}"
aa = f"{metrics.stats_accuracy_per_class(cm)[0]:.5f}"
iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"

print(oa)
print(aa)
print(iou)

#unique, counts = numpy.unique(pts, return_counts=True)
# print(unique)
# print(counts)