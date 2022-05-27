

import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
# python -m pip install --user git+https://github.com/DanielPollithy/pypcd.git

# ply file
def read_ply_with_parameters(path):
    '''
    :param path: path to .ply(onlu) file
    :return: return point cloud data in array
    '''
    plydata = PlyData.read(path)
    data = plydata.elements[0].data
    print(plydata.elements)
    data_pd = pd.DataFrame(data)
    data_np = np.zeros(data_pd.shape, dtype=float)
    property_names = data[0].dtype.names
    for i, name in enumerate(property_names):
        print(name) # print parameters
        data_np[:, i] = data_pd[name]
    return data_np[:,0:3],np.int64(data_np[:,3])



if __name__ == "__main__":
    #path = "../Toronto_3D/*.ply"
    #path = "../Toronto_3D/L001.ply"
    path = "/media/liangdao/DATA/origin_data/origin/area1.ply"
    train_point, train_labels = read_ply_with_parameters(path)
    point_path = path.replace(".ply",".txt")
    print(train_point)
    label_path = path.replace(".ply",".labels")
    np.savetxt(point_path, train_point,fmt='%1.6f')   # X is an array

    np.savetxt(label_path, train_labels.astype(int),fmt='%i')   # X is an array
    
    #show_point_clouds_in_points(path)