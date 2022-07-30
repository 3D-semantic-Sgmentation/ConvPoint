

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
    print("reading.....")
    plydata = PlyData.read(path)
    data = plydata.elements[0].data
    print(plydata.elements)
    data_pd = pd.DataFrame(data)
    data_np = np.zeros(data_pd.shape, dtype=float)
    property_names = data[0].dtype.names
    for i, name in enumerate(property_names):
        print(name) # print parameters
        data_np[:, i] = data_pd[name]
    data_np = data_np[data_np[:,-1]>0]
    values, counts = np.unique(data_np[:,-1], return_counts=True)
    print(values,counts)
    data_np[:,0] = data_np[:,0] - 627400.00
    data_np[:,1] = data_np[:,1] - 4842700.00
    return data_np[:,0:3],np.int64(data_np[:,-1])



if __name__ == "__main__":
    #path = "../Toronto_3D/*.ply"
    #path = "../Toronto_3D/L001.ply"
    base_root = "/media/liangdao/DATA/Toronto_3D/ConvPoint/"
    paths = [
        "L001.ply",  #[1. 2. 3. 4. 5. 6. 7. 8.] [11108072   389485  1403746  6027932   214652   263980  1724637    87631]
        "L001_1.ply",  #[0. 1. 2. 3. 4. 5. 6. 7. 8.] [ 265100 2977045  201842 1178373 2353800  155861  232780  209705   83262]
        "L001_2.ply",  #[0. 1. 2. 3. 4. 5. 6. 7. 8.] [  81937 8131027  187643  225373 3674132   58791   31200 1514932    4369]
        "L003_1.ply",   #[1. 2. 3. 4. 5. 6. 7. 8.] [10679842   261593   905132  2559628    76218    89175   901328   263969]
        "L003_2.ply",  # [1. 2. 3. 4. 5. 6. 7. 8.] [9882546  524026 1049604 9126145  261785  318815 1470708     294]
        "L002.ply",  # [1. 2. 3. 4. 5. 6. 7. 8.] [6305463  296140 1921653  883714   85041  154246  323003   18258]
        "L003.ply", # [1. 2. 3. 4. 5. 6. 7. 8.] [20562388   785619  1954736 11685773   338003   407990  2372036   264263]
        "L004.ply",
    ]
    for path in paths:
        train_point, train_labels = read_ply_with_parameters(base_root+path)
        point_path = base_root+path.replace(".ply",".txt")
        label_path = base_root+path.replace(".ply",".labels")
        np.savetxt(point_path, train_point,fmt='%1.6f')   # X is an array
        np.savetxt(label_path, train_labels.astype(int),fmt='%i')   # X is an array
    
    #show_point_clouds_in_points(path)