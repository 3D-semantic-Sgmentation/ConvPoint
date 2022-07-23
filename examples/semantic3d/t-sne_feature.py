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

