
import sys
sys.path.append('../../')

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import metrics as metrics
import pandas as pd
from dask.distributed import Client
import dask
import dask.dataframe as dd
from scipy.stats import norm, kurtosis,skew
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.stats import norm, kurtosis
import numpy as np
import scipy.stats

from matplotlib import lines
from matplotlib import patches
from matplotlib.patheffects import withStroke
from predata_tum import read_ply_with_parameters
from random import random
'''
Some important Result location
ALA with two domain loss:
/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-23-21-23-51/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-08-26-11-09-52
ALA With one domain loss
/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-23-21-23-51/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-26-18-45-01  16 blocksize
ALA with target loss
/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-10-03-05-08-53
ALA with target loss with DS:
/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop1.2_2022-10-01-22-34-13


Batch size
=4: /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-09-25-18-33-05
=8:    /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-09-21-08-43-17 
/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-09-22-17-50-52
=32: /media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-09-23-15-16-32

DS
/media/liangdao/DATA/segmentation/ConvPoint/data/SegBig_8192_nocolorTrue_drop0.5_2022-09-14-09-23-17
ALA+DS
/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-23-21-23-51/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-09-06-14-41-51
True Lable
/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud
No transfer
/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/results_TUM


ALD Domain Loss
/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_Domain_8192_nocolorTrue_drop0.5_2022-08-01-17-55-41/SegBig_Domain_8192_nocolorTrue_drop0.5_2022-08-02-16-12-18
'''




def show_image():
    #File_data = np.loadtxt("/media/liangdao/DATA/small/convpoint/train/pointcloud_txt/mls2016_8class_20cm_ascii_area1_voxels.txt", dtype=float)
    N_CLASSES=8
    cm = np.zeros((N_CLASSES, N_CLASSES))

    original_path = "/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud/" #"/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_Domain_8192_nocolorTrue_drop0.5_2022-08-01-17-55-41/SegBig_Domain_8192_nocolorTrue_drop0.5_2022-08-02-16-12-18/results_test0.8/"
    # original_path = "/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud/"
    original_files = ["mls2016_8class_20cm_ascii_area1","mls2016_8class_20cm_ascii_area2","mls2016_8class_20cm_ascii_area3"]
  
    semantic3d_path = "/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud/"
    semantic3d_files = ["domfountain_station3_xyz_intensity_rgb","sg27_station1_intensity_rgb","sg27_station5_intensity_rgb"]

    predict_root= "/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-23-21-23-51/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-08-25-18-44-10/results/"
    predict_file= ["mls2016_8class_20cm_ascii_area1_voxels.npy_pts.txt"]

    labels= []
    points =[]

    for i in range(len(original_files)):
        # point = np.loadtxt(original_path + original_files[i]+"_voxels.npy_pts.txt")
        # label = np.loadtxt(original_path + original_files[i]+"_voxels.labels")

        point = np.load(original_path + original_files[i]+"_voxels.npy")
        label = point[:,-1]-1
        point = point[:,0:3]

        values, counts = np.unique(label, return_counts=True)
        print(values)
        print(counts)
        # point,label = read_ply_with_parameters(original_path+ original_files[i]+".ply")
        # point = np.loadtxt(no_transfer_path+ no_transfer_files[i])
        # points = np.concatenate(points,point)
        
        # if points == []:
        #     points = point
        #     labels = label
        # else:
        #     points = np.concatenate((points, point), axis=0)
        #     labels = np.concatenate((labels, label), axis=0)
        points = point
        labels = label
        print(points.shape)
        print(labels.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        
        max_label = 10
        colors = plt.get_cmap("tab20")(labels/(max_label if max_label > 0 else 1))
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
        o3d.visualization.draw_geometries([pcd])

    # legend_color =  plt.get_cmap("tab20")(np.array([0,1,2,3,4,5,6,7])/(max_label if max_label > 0 else 1))
    # print(legend_color)

    # lo = plt.scatter(random(), random(), marker='o', color=legend_color[0])
    # ll = plt.scatter(random(), random(), marker='o', color=legend_color[1])
    # l  = plt.scatter(random(), random(), marker='o', color=legend_color[2])
    # a  = plt.scatter(random(), random(), marker='o', color=legend_color[3])
    # h  = plt.scatter(random(), random(), marker='o', color=legend_color[4])
    # hh = plt.scatter(random(), random(), marker='o', color=legend_color[5])
    # ho = plt.scatter(random(), random(), marker='o', color=legend_color[6])
    # hl = plt.scatter(random(), random(), marker='o', color=legend_color[7])

    # plt.legend((lo, ll, l, a, h, hh, ho, hl),
    #         ("Man-made terrain", "Natural terrain","High vegetation", "Low vegetation", "Buildings", "Hard scape", "Scanning artefacts", "Vehicles"),
    #         scatterpoints=1,
    #         loc='lower left',
    #         ncol=1,
    #         fontsize=10)

    # plt.show()

def draw_figure(values, counts, area):
    max_label = 10

    count_max = max(counts)
    print(count_max)
    print(count_max // 10**(len(str(count_max))-1))
    count_max = ((count_max // 10**(len(str(count_max))-1))+1)*10**(len(str(count_max))-1)
    print(count_max)

    bars = ["man-made terrain", "natural terrain","high vegetation", "low vegetation", "buildings", "hard scape", "scanning artefacts", "cars"]
    colors = plt.get_cmap("tab20")(values/(max_label if max_label > 0 else 1))    
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.barh(values, counts, height=0.55, align="edge", color=colors)
    
    PAD = 0.05*count_max
    for name, count, y_pos in zip(bars, counts, values):
        x = 0
        color = "white"
        path_effects = None
        print(y_pos)
        if count < 0.2*count_max:
            x = count
            color = colors.tolist()[values.tolist().index(y_pos)]     # y-1
            path_effects=[withStroke(linewidth=6, foreground="white")]
        
        ax.text(
            x + PAD, y_pos + 0.5 / 2, bars[int(y_pos)-1], 
            color=color, fontfamily="Econ Sans Cnd", fontsize=18, va="center",
            path_effects=path_effects
        ) 

    ax.xaxis.set_ticks([i for i in range(0, count_max,count_max//10)])
    ax.xaxis.set_ticklabels([i for i in range(0, count_max, count_max//10)], size=16, fontfamily="Econ Sans Cnd", fontweight=100)
    ax.xaxis.set_tick_params(labelbottom=False, labeltop=True, length=0)

    ax.set_xlim((0, count_max))
    ax.set_ylim((0, len(bars)+2))

    # Set whether axis ticks and gridlines are above or below most artists.
    ax.set_axisbelow(True)
    ax.grid(axis = "x", color="#A8BAC4", lw=1.2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_lw(1.5)
    # This capstyle determines the lines don't go beyond the limit we specified
    # see: https://matplotlib.org/stable/api/_enums_api.html?highlight=capstyle#matplotlib._enums.CapStyle
    ax.spines["left"].set_capstyle("butt")

    # Hide y labels
    ax.yaxis.set_visible(False)
    ax.legend()
    # Make room on top and bottom
    # Note there's no room on the left and right sides
    fig.subplots_adjust(left=0.005, right=1, top=0.8, bottom=0.1)

    # Add title
    fig.text(
    0, 0.925, "TUM-MLS Area "+str(area+1), 
    #0, 0.925, "TUM-MLS", 
    fontsize=22, fontweight="bold", fontfamily="Econ Sans Cnd"
    )
    # Add subtitle
    fig.text(
    0, 0.875, "Number of the scene descriptor", 
    fontsize=20, fontfamily="Econ Sans Cnd"
    )

    # Add caption
    #source = "Sources: Semantic3D (Hackel et al., 2017))"
    source = "Sources: TUM-MLS (Zhu et al., 2020),"
    fig.text(
    0, 0.04, source, color="grey", 
    fontsize=14, fontfamily="Econ Sans Cnd"
    )


    # Add line and rectangle on top.
    fig.add_artist(lines.Line2D([0, 1], [1, 1], lw=3, color="red", solid_capstyle="butt"))
    fig.add_artist(patches.Rectangle((0, 0.975), 0.05, 0.025, color="red"))

    # Set facecolor, useful when saving as .png
    fig.set_facecolor("white")
    fig.savefig("/media/liangdao/DATA/Paper/TUM_"+str(area+1)+"_histogram.png", dpi=300)
    plt.show()


def statistic_class():

    # original_path = "/media/liangdao/DATA/Semantic3D/"
    # original_files = [ "bildstein_station1_xyz_intensity_rgb_voxels.npy",
    #         "bildstein_station3_xyz_intensity_rgb_voxels.npy",
    #         "bildstein_station5_xyz_intensity_rgb_voxels.npy",
    #         "domfountain_station1_xyz_intensity_rgb_voxels.npy",
    #         "domfountain_station2_xyz_intensity_rgb_voxels.npy",
    #         "domfountain_station3_xyz_intensity_rgb_voxels.npy",
    #         "neugasse_station1_xyz_intensity_rgb_voxels.npy",
    #         "sg27_station1_intensity_rgb_voxels.npy",
    #         "sg27_station2_intensity_rgb_voxels.npy",
    #         "sg27_station4_intensity_rgb_voxels.npy",
    #         "sg27_station5_intensity_rgb_voxels.npy",
    #         "sg27_station9_intensity_rgb_voxels.npy",
    #         "sg28_station4_intensity_rgb_voxels.npy",
    #         "untermaederbrunnen_station3_xyz_intensity_rgb_voxels.npy",
    #         "untermaederbrunnen_station1_xyz_intensity_rgb_voxels.npy",
    #         ]
    original_path = "/media/liangdao/DATA/small/area123/"
    original_files = ["mls2016_8class_20cm_ascii_area1_voxels.npy","mls2016_8class_20cm_ascii_area2_voxels.npy","mls2016_8class_20cm_ascii_area3_voxels.npy"]
    
    labels = []
    for i in range(len(original_files)):
        print(i)
        File_data = np.loadtxt(original_path+original_files[i].replace("_voxels.npy",".labels"), dtype=float)
        #labels.extend(File_data.data.tolist())
        values, counts = np.unique(File_data.data, return_counts=True)
        print(values,counts/sum(counts))
        draw_figure(values, counts, i)



    
def statistic_iou():
    N_CLASSES=8
    cm = np.zeros((N_CLASSES, N_CLASSES))

    original_path = "/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud/"
    original_files = ["mls2016_8class_20cm_ascii_area1_voxels.npy","mls2016_8class_20cm_ascii_area2_voxels.npy","mls2016_8class_20cm_ascii_area3_voxels.npy"]
    original_files = [ "bildstein_station1_xyz_intensity_rgb_voxels.npy",
            "bildstein_station3_xyz_intensity_rgb_voxels.npy",
            "domfountain_station1_xyz_intensity_rgb_voxels.npy",
            "domfountain_station3_xyz_intensity_rgb_voxels.npy",
            "neugasse_station1_xyz_intensity_rgb_voxels.npy",
            "sg27_station1_intensity_rgb_voxels.npy",
            "sg27_station5_intensity_rgb_voxels.npy",
            "untermaederbrunnen_station1_xyz_intensity_rgb_voxels.npy",
            "bildstein_station5_xyz_intensity_rgb_voxels.npy",
            "domfountain_station2_xyz_intensity_rgb_voxels.npy",
            "sg27_station4_intensity_rgb_voxels.npy",
            "sg27_station2_intensity_rgb_voxels.npy",
            "sg27_station9_intensity_rgb_voxels.npy",
            "sg28_station4_intensity_rgb_voxels.npy",
            "untermaederbrunnen_station3_xyz_intensity_rgb_voxels.npy",]
    # predict_root= "/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-10-03-05-08-53"+"/results/"
    predict_root = "/media/liangdao/DATA/segmentation/ConvPoint/data/SEMANTIC3D/SegBig_nocolor/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-07-23-21-23-51/SegBig_GAN_8192_nocolorTrue_drop0.5_2022-08-25-18-44-10/results/"

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
    oa = f"{metrics.stats_overall_accuracy(cm):.5f}"
    print(iou)
    print(metrics.stats_iou_per_class(cm)[1])
    print("OA",oa)

def statistic_similarity():
    similarity = 0
    semantic_path = "/media/liangdao/DATA/Semantic3D/"
    semantic_files = [ 
            # "bildstein_station1_xyz_intensity_rgb.txt",
            # "bildstein_station3_xyz_intensity_rgb.txt",
            # "domfountain_station1_xyz_intensity_rgb.txt",
            "domfountain_station3_xyz_intensity_rgb.txt",
            # "neugasse_station1_xyz_intensity_rgb.txt",
            # "sg27_station1_intensity_rgb.txt",
            # "sg27_station5_intensity_rgb.txt",
            # "untermaederbrunnen_station1_xyz_intensity_rgb.txt",
            # "bildstein_station5_xyz_intensity_rgb.txt",
            # "domfountain_station2_xyz_intensity_rgb.txt",
            # "sg27_station4_intensity_rgb.txt",
            # "sg27_station2_intensity_rgb.txt",
            # "sg27_station9_intensity_rgb.txt",
            # "sg28_station4_intensity_rgb.txt",
            # "untermaederbrunnen_station3_xyz_intensity_rgb.txt"
            ]
    

    # std mean
    # (29697591, 7)
    # bildstein_station1_xyz_intensity_rgb.txt [] [ 1.35460013  1.75404757 -0.47692838]
    # (23995481, 7)
    # bildstein_station3_xyz_intensity_rgb.txt [] [ 0.9319305  -0.94795691 -0.63205596]
    # (44990641, 7)
    # domfountain_station1_xyz_intensity_rgb.txt [ ] [-5.01035378 -4.09153273  1.37995864]
    # (35207289, 7)
    # domfountain_station3_xyz_intensity_rgb.txt [ ] [-0.02734547  0.21145913  1.68909315]
    # (50122464, 7)
    # neugasse_station1_xyz_intensity_rgb.txt [] [ 0.76424399 -1.33139415  1.2634485 ]
    # (322088562, 7)
    # sg27_station1_intensity_rgb.txt [  546.049890 41.846564 38.652644 39.842828] []
    # sg27_station5_intensity_rgb.txt [ 500.272786 49.924822 44.766986 47.472902][]
    # untermaederbrunnen_station1_xyz_intensity_rgb.txt [ 1040.604481 68.323981 71.939291 73.389681][]
    # bildstein_station5_xyz_intensity_rgb.txt [ 389.451950 42.447659 43.898096 42.784511]
    # domfountain_station2_xyz_intensity_rgb.txt [ 386.369761 48.378147 49.432622 52.170100][]
    # sg27_station4_intensity_rgb.txt [ 633.512334 49.584507 46.316924 50.057193] []
    # sg27_station2_intensity_rgb.txt [ 754.632151 47.825694 42.891405  45.071711][]
    # sg27_station9_intensity_rgb.txt [ 661.799544 48.770988 45.226677 46.328906] []
    # sg28_station4_intensity_rgb.txt [ 516.762784 36.692833 33.022873 35.692124]
    # untermaederbrunnen_station3_xyz_intensity_rgb.txt [ 1502.231723 60.382947 64.802428 66.839425][]



    # mls2016_8class_20cm_ascii_area1.ply [] [ 11.72682783 -64.93507978 -18.53635253]
    # mls2016_8class_20cm_ascii_area2.ply [38.35589287 77.89695674  6.02940227] [-79.35250704 -32.18667826 -18.78095372]
    # mls2016_8class_20cm_ascii_area3.ply [43.63160632 89.09122201  5.82104104] [-168.14442014    7.67007477  -19.63899707]


    tummls_path = "/media/liangdao/DATA/small/area123/"
    tummls_files = ["mls2016_8class_20cm_ascii_area1.ply","mls2016_8class_20cm_ascii_area2.ply","mls2016_8class_20cm_ascii_area3.ply"]


    print("start")

    # mean, standard deviation, skewness, kurtosis, Hellinger distance
    for i in range(0, len(semantic_files)):
        # semantic_data = np.loadtxt(semantic_path+semantic_files[i])
        
        semantic_data = pd.read_csv(semantic_path+semantic_files[i],sep=";|:|,| ")
   
        print(semantic_data.shape)
        # for chunk in semantic_data:
        #     print(chunk)
        #     # print(chunk.sum(axis=0))
        #     total += chunk.sum(axis=0)  # Cannot broadcast np.ndarray with operand of type <class 'list'>
        #     count += chunk.shape[0]
        #     print(total)
        #     print(count)
        # print(semantic_data.shape)
        # semantic_point = semantic_data[:,0:3]
        print(semantic_files[i],semantic_data.min(axis=0))

    # for i in range(len(tummls_files)):
    #     #File_data = np.loadtxt(tummls_path+tummls_files[i], dtype=float)
    #     pcd = o3d.io.read_point_cloud(tummls_path+tummls_files[i]) # Read the point cloud
    #     tummls_point = np.asarray(pcd.points) 
        
    #     print(tummls_files[i]) # np.std(tummls_point,axis=0),np.mean(tummls_point,axis=0)
    #     print(np.mean(tummls_point,axis=0))

    #return similarity


def dask_statistic_similarity():
    client = Client(n_workers=3)
    print("start")
    semantic_path = "/media/liangdao/DATA/Semantic3D/"
    semantic_files = [ 
            # "bildstein_station1_xyz_intensity_rgb.txt",
            # "bildstein_station3_xyz_intensity_rgb.txt",
            # "domfountain_station1_xyz_intensity_rgb.txt",
            # "domfountain_station3_xyz_intensity_rgb.txt",
            # "neugasse_station1_xyz_intensity_rgb.txt",
            # "sg27_station1_intensity_rgb.txt",
            # "sg27_station5_intensity_rgb.txt",
            # "untermaederbrunnen_station1_xyz_intensity_rgb.txt",
            # "bildstein_station5_xyz_intensity_rgb.txt",
            # "domfountain_station2_xyz_intensity_rgb.txt",
            # "sg27_station4_intensity_rgb.txt",
            "sg27_station2_intensity_rgb.txt",
            "sg27_station9_intensity_rgb.txt",
            "sg28_station4_intensity_rgb.txt",
            "untermaederbrunnen_station3_xyz_intensity_rgb.txt"
            ]
    for semantic_file in semantic_files:
        ddf = dd.read_csv(semantic_path+semantic_file,header=None,sep=";|:|,| ",engine='python')
        count = ddf.count(axis = 0)
        # print(semantic_file)
        print(ddf.head())
        #skew = ddf.skew(axis = 0)
        #max_num = ddf.max(axis = 0)
        # min_num = ddf.min(axis = 0)
        #std = ddf.std(axis=0)
        #sumary = ddf.sum(axis=0)
        #mean = ddf.mean(axis=0)
        #result = std.compute()
        kurtosis_value = kurtosis(ddf)
        #print(semantic_file,skew.compute(),max_num.compute(),min_num.compute())
        #print(semantic_file, kurtosis.compute())
        print(semantic_file, kurtosis_value)


def statistic_result():
    
    # calculate Spearman's correlation
    lidar_features = pd.read_csv("LIDAR.csv")
    lidar_features = lidar_features[["std_x","std_y","std_z","skew_x","skew_y","skew_z","meanz_minz"]]
    # print(lidar_features["mean_z"]-lidar_features["min_z"])
    result = {}
    for i in range(len(lidar_features)-3):
        corr=0
        for j in range(1,4):
            #cos_sim = cosine_similarity(lidar_features.iloc[i], lidar_features.iloc[-1])
            _corr, _ = spearmanr(lidar_features.iloc[i], lidar_features.iloc[-j])
            corr+=_corr
        result[lidar_features.index[i]] = corr

    print(dict(sorted(result.items(), key=lambda item: item[1])))


# Importing libraries
  
import numpy as np
import pandas as pd
import scipy as stats
from scipy.stats import chi2
import os
from scipy.spatial.distance import pdist
import seaborn as sns
import seaborn as sns

# calculateMahalanobis Function to calculate
# the Mahalanobis distance
def calculateMahalanobis(y=None, data=None, cov=None):
  
    y_mu = y - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()

def distance():


    folder = "/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud"
    filelist = [
        "bildstein_station1_xyz_intensity_rgb_voxels.npy",
        "bildstein_station3_xyz_intensity_rgb_voxels.npy",
        "bildstein_station5_xyz_intensity_rgb_voxels.npy",
        "domfountain_station1_xyz_intensity_rgb_voxels.npy",
        "domfountain_station2_xyz_intensity_rgb_voxels.npy",
        "domfountain_station3_xyz_intensity_rgb_voxels.npy",
        "untermaederbrunnen_station1_xyz_intensity_rgb_voxels.npy",
        "untermaederbrunnen_station3_xyz_intensity_rgb_voxels.npy",
        "neugasse_station1_xyz_intensity_rgb_voxels.npy",
        "sg27_station1_intensity_rgb_voxels.npy",
        "sg27_station2_intensity_rgb_voxels.npy",
        "sg27_station4_intensity_rgb_voxels.npy",
        "sg27_station5_intensity_rgb_voxels.npy",  
        "sg27_station9_intensity_rgb_voxels.npy",
        "sg28_station4_intensity_rgb_voxels.npy",

    ]
    
    original_path =  "/media/liangdao/DATA/segmentation/ConvPoint/data/Prepare/train/pointcloud"  # "/media/liangdao/DATA/Paris_and_Lille/train/pointcloud/"
    #["Lille1_1_voxels.npy","Lille1_2_voxels.npy","Lille2_voxels.npy","Paris_voxels.npy"]
    target_filelist= [
        "mls2016_8class_20cm_ascii_area1_voxels.npy",
        "mls2016_8class_20cm_ascii_area2_voxels.npy",
        "mls2016_8class_20cm_ascii_area3_voxels.npy",
        # "L001_voxels.npy",
        # "L002_voxels.npy",
        # "L003_1_voxels.npy",
        # "L003_2_voxels.npy",
        # "L004_voxels.npy"
    ]  
    

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range   
    # get the features
    his_source = []

    sns.set_style('whitegrid')
    fig, axs = plt.subplots(5,3)
    plt.ticklabel_format(style='plain', axis='y',useOffset=False)
    for i in range(0,len(filelist)):
        print(i//3, i%3)
        pts = np.load(os.path.join(folder, filelist[i]))
        pts = pts[:,2]
        print(pts.shape)
        # his = axs[i//5, i%5].hist(pts)
        # his_source.append(normalization(his[0]))
        his = sns.histplot(ax=axs[i//3, i%3], data=pts, element="poly", fill=False, common_norm=True,bins=10, kde=True)
        his.set(xlabel=None)
        his.set(ylabel=None)
        his = sns.histplot(ax=axs[i//3, i%3], data=pts, common_norm=True,bins=10, kde=True)
        his.set(xlabel=None)
        his.set(ylabel=None)
        
        #his_source.append(normalization(his[0]))
        axs[i//3, i%3].set_title(filelist[i].replace("_intensity_rgb_voxels.npy",""))
        axs[i//3, i%3].set_xticks([])
    

    fig.text(0.5, 0.04, 'Height', ha='center',fontsize=15)
    fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical',fontsize=15)
    plt.show()

    fig, axs = plt.subplots(len(target_filelist))
    plt.ticklabel_format(style='plain', axis='y',useOffset=False)
    his_target = []
    for i in range(0,len(target_filelist)):
        plt.xticks([])
        q = np.load(os.path.join(original_path, target_filelist[i]))
        q = q[:,2]
        print(q.shape)

        # his = axs[i].hist(q)
        # his_target.append(normalization(his[0]))
        his = sns.histplot(ax=axs[i], data=q, kde=True,bins=10, common_norm=True,element="poly", fill=False)
        his.set(xlabel=None)
        his.set(ylabel=None)
        his = sns.histplot(ax=axs[i], data=q, common_norm=True, kde=True,bins=10)
        his.set(xlabel=None)
        his.set(ylabel=None)
        #his_target.append(normalization(his[0]))
        axs[i].set_title(target_filelist[i].replace("_voxels.npy","").replace("2016_8class_20cm_ascii",""))
        axs[i].set_xticks([])
    
    fig.text(0.5, 0.04, 'Height', ha='center',fontsize=15)
    fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical',fontsize=15)
    
    plt.show()
    print(his_source)
    print(his_target)
    result={}
    for i in range(len(his_source)):
        corr=0
        
        for j in range(len(his_target)):
            #cos_sim = cosine_similarity(lidar_features.iloc[i], lidar_features.iloc[-1])
            X=np.vstack([his_source[i],his_target[j]])
            d2=1-pdist(X,'cosine')
            # _corr, _ = spearmanr(his_source[i], his_target[j])
            corr+=d2
            
   
        result[filelist[i].replace("_intensity_rgb_voxels.npy","")] = corr
    
    print(dict(sorted(result.items(), key=lambda item: item[1])))

# {'bildstein_station1_xyz': 0.6727272727272727, 'untermaederbrunnen_station1_xyz': 1.2909090909090908, 'sg27_station4': 1.6424242424242426, 'bildstein_station3_xyz': 1.945454545454545, 'sg27_station5': 2.3658976412781243, 'untermaederbrunnen_station3_xyz': 2.3818181818181814, 'domfountain_station3_xyz': 2.818181818181818, 'bildstein_station5_xyz': 2.8666666666666663, 'sg28_station4': 2.915151515151515, 'sg27_station1': 2.927272727272727, 'domfountain_station1_xyz': 2.9636363636363634, 'sg27_station9': 2.9636363636363634, 'domfountain_station2_xyz': 2.9757575757575756, 'neugasse_station1_xyz': 2.9757575757575756, 'sg27_station2': 2.9757575757575756}

    #plt.show()



if __name__ == '__main__': 
    # show_image()
    # dask_statistic_similarity()
    statistic_iou()