import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import scipy as stats
from scipy.stats import chi2
import os
from scipy.spatial.distance import pdist
import seaborn as sns
import seaborn as sns




def distance():

    stat = "percent"  # frequency|probability|percent|density
    folder = "/media/liangdao/DATA/MyConv/data/train/pointcloud"


    original_path = "/media/liangdao/DATA/origin_data/origin/train/pointcloud"

    target_filelist= [

        "area1_voxels.npy",

    ]  

    config = {
        "font.size": 15,
        "mathtext.fontset":'stix',
        }
    rcParams.update(config)
    fig, axs = plt.subplots(1)
    plt.ticklabel_format(style='plain', axis='y',useOffset=True)
    # his_target = []
    tum_height = []
    # len(target_filelist)
    for i in range(0,1):
        q=np.load(os.path.join(original_path, target_filelist[i]))
        q = q[:,2]
        tum_height.extend(q)
        print(len(tum_height))
        # his = axs[i].hist(q)
        # his_target.append(normalization(his[0]))
        # plt.xticks([])

    print("1")
    his = sns.histplot(ax=axs, data=tum_height, kde=False, bins=10, common_norm=True, element="poly", stat=stat,fill=False)
    his = sns.histplot(ax=axs, data=tum_height, common_norm=True, stat=stat,bins=10) # kde=True,
    his.set(xlabel=None)
    his.set(ylabel=None)
    axs.set_title("TUM-MLS",fontsize=15)
    # axs[0].set_xticks([])

    fig.text(0.5, 0.04, 'Height(m)', ha='center') #, fontfamily="Times New Roman"
    fig.text(0.04, 0.5, 'Percent(%)', va='center', rotation='vertical')
    axs.tick_params(axis='x')
    axs.tick_params(axis='y')
    # matplotlib.rc('axes', titlesize=15)
    # plt.rc('font', size=15) 
    plt.title(r'$\mathrm{Times \; New \; Roman}\/\/ $')

    plt.show()


# {'bildstein_station1_xyz': 0.6727272727272727, 'untermaederbrunnen_station1_xyz': 1.2909090909090908, 'sg27_station4': 1.6424242424242426, 'bildstein_station3_xyz': 1.945454545454545, 'sg27_station5': 2.3658976412781243, 'untermaederbrunnen_station3_xyz': 2.3818181818181814, 'domfountain_station3_xyz': 2.818181818181818, 'bildstein_station5_xyz': 2.8666666666666663, 'sg28_station4': 2.915151515151515, 'sg27_station1': 2.927272727272727, 'domfountain_station1_xyz': 2.9636363636363634, 'sg27_station9': 2.9636363636363634, 'domfountain_station2_xyz': 2.9757575757575756, 'neugasse_station1_xyz': 2.9757575757575756, 'sg27_station2': 2.9757575757575756}

    #plt.show()



if __name__ == '__main__': 
    # show_image()
    # dask_statistic_similarity()
    distance()
