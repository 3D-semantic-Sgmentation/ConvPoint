# Semantic3D Example with ConvPoint

# add the parent folder to the python path to access convpoint library
import sys
sys.path.append('../../')
import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors

from utils import metrics as metrics

# from convpoint.knn.lib.python import nearest_neighbors as nearest_neighbors

import numpy as np
import argparse
from datetime import datetime
import os
import random
from tqdm import tqdm
import itertools    
import time


import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms

from sklearn.metrics import confusion_matrix
import time
from torch.utils.data import TensorDataset, ConcatDataset
import logging
from torch.optim import lr_scheduler
# import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors

from PIL import Image
torch.cuda.empty_cache()
import gc

from torch_poly_lr_decay import PolynomialLRDecay


gc.collect()
torch.cuda.memory_summary(device=None, abbreviated=False)
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated(device="cuda:0"))
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC

def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    print(pts_dest)
    indices = nearest_neighbors.knn(pts_src.copy(), pts_dest.copy(), K, omp=True)
    print(indices)
    if K==1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1],])
    return np.dot(batch_data, rotation_matrix)

def discrepancy(out1, out2):
    """discrepancy loss"""
    out = torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))
    return out


# Part dataset only for training / validation
class PartDataset():

    def __init__ (self, filelist, folder,
                    training=False, 
                    iteration_number = None,
                    block_size=8,
                    npoints = 8192,
                    nocolor=True,
                    transfer=False):

        self.folder = folder
        self.training = training
        self.filelist = filelist
        self.bs = block_size
        self.nocolor = nocolor

        self.npoints = npoints
        self.iterations = iteration_number
        self.verbose = False
        self.transfer = transfer

        
        self.transform = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4)

    def __getitem__(self, index):
        
        # load the data
        index = random.randint(0, len(self.filelist)-1)
        pts = np.load(os.path.join(self.folder, self.filelist[index]))

        # get the features
        fts = pts[:,3:6]

        # get the labels
        lbs = pts[:, 6].astype(int)-1 # the generation script label starts at 1

        # get the point coordinates
        pts = pts[:, :3]


        # pick a random point
        pt_id = random.randint(0, pts.shape[0]-1)
        pt = pts[pt_id]

        # create the mask
        mask_x = np.logical_and(pts[:,0]<pt[0]+self.bs/2, pts[:,0]>pt[0]-self.bs/2)
        mask_y = np.logical_and(pts[:,1]<pt[1]+self.bs/2, pts[:,1]>pt[1]-self.bs/2)
        mask = np.logical_and(mask_x, mask_y)
        pts = pts[mask]
        lbs = lbs[mask]
        fts = fts[mask]
        
        # random selection
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]
        lbs = lbs[choice]
        fts = fts[choice]

        # data augmentation
        if self.training:
            # random rotation
            pts = rotate_point_cloud_z(pts)

            # random jittering
            fts = fts.astype(np.uint8)
            fts = np.array(self.transform( Image.fromarray(np.expand_dims(fts, 0)) ))
            fts = np.squeeze(fts, 0)
        
        fts = fts.astype(np.float32)
        fts = fts / 255 - 0.5

        if self.nocolor:
            fts = np.ones((pts.shape[0], 1))

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        lbs = torch.from_numpy(lbs).long()

        # if self.transfer==True:
        #     clabel = torch.from_numpy(np.zeros(lbs.shape[0])).float()
        # if self.transfer==False:
        #     clabel = torch.from_numpy(np.ones(lbs.shape[0])).float()

        return pts, fts, lbs

    def __len__(self):
        return self.iterations

class PartDatasetTest():

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:,0]<pt[0]+bs/2, self.xyzrgb[:,0]>pt[0]-bs/2)
        mask_y = np.logical_and(self.xyzrgb[:,1]<pt[1]+bs/2, self.xyzrgb[:,1]>pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__ (self, filename, folder,
                    block_size=8,
                    npoints = 8192,
                    test_step=0.8, nocolor=True):

        self.folder = folder
        self.bs = block_size
        self.npoints = npoints
        self.verbose = False
        self.nocolor = nocolor
        self.filename = filename

        # load the points
        self.xyzrgb = np.load(os.path.join(self.folder, self.filename))
        step = test_step
        discretized = ((self.xyzrgb[:,:2]).astype(float)/step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float)*step

    def __getitem__(self, index):
        # index = random.randint(0, len(self.pts)-1)
        # get the data
        mask = self.compute_mask(self.pts[index], self.bs)
        pts = self.xyzrgb[mask]

        # choose right number of points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]

        # labels will contain indices in the original point cloud
        lbs = np.where(mask)[0][choice]

        # separate between features and points
        if self.nocolor:
            fts = np.ones((pts.shape[0], 1))
        else:
            # fts = pts[:,3:6]
            fts = np.zeros((pts.shape[0], 3))
            fts[:,2]=np.ones((pts.shape[0]))
            fts = fts.astype(np.float32)
            fts = fts / 255 - 0.5

        pts = pts[:, :3].copy()

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs

    def __len__(self):
        return len(self.pts)


def get_model(model_name, input_channels, output_channels, args):
    if model_name == "SegBig":
        from networks.network_seg import SegBig as Net
        return Net(input_channels, output_channels, args=args)
    elif model_name == "SegSmall":
        from networks.network_seg import SegSmall as Net
        return Net(input_channels, output_channels)
    elif model_name == "SegBig_GAN":
        from networks.network_seg import SegBig_FG as FGNet
        from networks.network_seg import SegBig_Dis as DisNet
        return FGNet(input_channels, output_channels), DisNet(input_channels, output_channels)
    elif model_name == "Gan":
        from networks.network_seg import SegSmall_Features_Generator as GenNet
        from networks.network_seg import SegSmall_Features_Discriminotor as DisNet
        return GenNet(input_channels, output_channels), DisNet(input_channels, output_channels)
        
    
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', '-s', help='Path to data folder')
    parser.add_argument("--savedir", type=str, default="./results")
    parser.add_argument('--block_size', help='Block size', type=float, default=16)  # thing to change
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", "-b", type=int, default=10)
    parser.add_argument("--iter", "-i", type=int, default=1200)
    parser.add_argument("--npoints", "-n", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--nocolor",default=True)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--continuetrain", action="store_true")
    parser.add_argument("--finetuning", action="store_true")
    parser.add_argument("--test_step", default=0.8, type=float)
    parser.add_argument("--model", default="SegBig_GAN", type=str)
    parser.add_argument("--drop", default=0.5, type=float)
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta", type=float, default=0.001, help="adam: decay of first order momentum of gradient")
    
    args = parser.parse_args()
    
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    root_folder = os.path.join(args.savedir, "{}_{}_nocolor{}_drop{}_{}".format(
            args.model, args.npoints, args.nocolor, args.drop, time_string))

    filelist_train=[
        "mls2016_8class_20cm_ascii_area1_voxels.npy",
        "mls2016_8class_20cm_ascii_area2_voxels.npy",
        "mls2016_8class_20cm_ascii_area3_voxels.npy",
    ]
    filelist_train_trans=[
        "bildstein_station1_xyz_intensity_rgb_voxels.npy",
        "bildstein_station3_xyz_intensity_rgb_voxels.npy",
        "bildstein_station5_xyz_intensity_rgb_voxels.npy",
        "domfountain_station1_xyz_intensity_rgb_voxels.npy",
        "domfountain_station2_xyz_intensity_rgb_voxels.npy",
        "domfountain_station3_xyz_intensity_rgb_voxels.npy",
        "neugasse_station1_xyz_intensity_rgb_voxels.npy",
        "untermaederbrunnen_station1_xyz_intensity_rgb_voxels.npy",
        "untermaederbrunnen_station3_xyz_intensity_rgb_voxels.npy",
        "sg27_station1_intensity_rgb_voxels.npy",
        "sg27_station2_intensity_rgb_voxels.npy",
        "sg27_station5_intensity_rgb_voxels.npy",  
        "sg27_station4_intensity_rgb_voxels.npy",
        "sg27_station9_intensity_rgb_voxels.npy",
        "sg28_station4_intensity_rgb_voxels.npy",
    ]
    

    filelist_val=[
        "mls2016_8class_20cm_ascii_area1_voxels.npy",
        "mls2016_8class_20cm_ascii_area2_voxels.npy",
        "mls2016_8class_20cm_ascii_area3_voxels.npy",
    ]
    filelist_test=[
        # "mls2016_8class_20cm_ascii_area1_voxels.npy",
        # "mls2016_8class_20cm_ascii_area2_voxels.npy",
        # "mls2016_8class_20cm_ascii_area3_voxels.npy",
        "bildstein_station1_xyz_intensity_rgb_voxels.npy",
        "bildstein_station3_xyz_intensity_rgb_voxels.npy",
        "bildstein_station5_xyz_intensity_rgb_voxels.npy",
        "domfountain_station1_xyz_intensity_rgb_voxels.npy",
        "domfountain_station2_xyz_intensity_rgb_voxels.npy",
        "domfountain_station3_xyz_intensity_rgb_voxels.npy",
        "neugasse_station1_xyz_intensity_rgb_voxels.npy",
        "untermaederbrunnen_station1_xyz_intensity_rgb_voxels.npy",
        "untermaederbrunnen_station3_xyz_intensity_rgb_voxels.npy",
        "sg27_station1_intensity_rgb_voxels.npy",
        "sg27_station2_intensity_rgb_voxels.npy",
        "sg27_station5_intensity_rgb_voxels.npy",  
        "sg27_station4_intensity_rgb_voxels.npy",
        "sg27_station9_intensity_rgb_voxels.npy",
        "sg28_station4_intensity_rgb_voxels.npy",
    ]
    print(filelist_train,filelist_train_trans)
    print(filelist_val)

    N_CLASSES= 8

    saved_args = locals()
    print(saved_args)
    # create model
    print("Creating the network...", end="", flush=True)
    if args.nocolor:
        FGNet, dis = get_model(args.model, input_channels=1, output_channels=N_CLASSES, args=args)
    else:
        FGNet, dis = get_model(args.model, input_channels=3, output_channels=N_CLASSES, args=args)

    if args.test:
        FGNet.load_state_dict(torch.load(os.path.join(args.savedir, "FGNet_state_dict.pth")))
        dis.load_state_dict(torch.load(os.path.join(args.savedir, "dis_state_dict.pth")))
        print("test")
    if args.continuetrain:
        FGNet.load_state_dict(torch.load(os.path.join(args.savedir, "FGNet_state_dict.pth")))
        dis.load_state_dict(torch.load(os.path.join(args.savedir, "dis_state_dict.pth")))
        print("loaded model")
    if args.finetuning:
        # ---------------------
        #  Load pretrained model
        # --------------------
        pretrained_dict = torch.load(os.path.join(args.savedir, "state_dict.pth"))

        FGNet_dict = FGNet.state_dict()

        pretrained_dict_gen = {k: v for k, v in pretrained_dict.items() if k in FGNet_dict}
        # 2. overwrite entries in the existing state dict
        FGNet_dict.update(pretrained_dict_gen)

        FGNet.load_state_dict(FGNet_dict)
        print("start load dis model")

        dis_dict = dis.state_dict()
        for k, v in pretrained_dict.items():
            if k in dis_dict:
                print(k)
        # 1. filter out unnecessary keys
        pretrained_dict_dis = {k: v for k, v in pretrained_dict.items() if k in dis_dict}
        # 2. overwrite entries in the existing state dict
        dis_dict.update(pretrained_dict_dis)
        dis.load_state_dict(dis_dict)

        # dis2.load_state_dict(dis_dict)
        print("load pretrained models")
        
    FGNet.cuda()
    dis.cuda()

    print("Done")
    print("discriminator output 1 class(Linear)")
    print("Gan Model")




    ##### TRAIN
    if not args.test:
        print("Create the datasets...", end="", flush=True)

        ds = PartDataset(filelist_train, args.rootdir,
                                training=True, block_size=args.block_size,
                                iteration_number=args.batch_size*args.iter,  #16000
                                npoints=args.npoints,
                                nocolor=args.nocolor,
                                transfer=False)

        ds_transfer = PartDataset(filelist_train_trans, args.rootdir,
                                training=True, block_size=args.block_size,
                                iteration_number=args.batch_size*args.iter,
                                npoints=args.npoints,
                                nocolor=args.nocolor,
                                transfer=True)
        
        train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.threads)

        train_trans_loader = torch.utils.data.DataLoader(ds_transfer, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.threads)
        val = PartDataset(filelist_val, args.rootdir,
                                training=True, block_size=args.block_size,
                                iteration_number=args.batch_size*args.iter,
                                npoints=args.npoints,
                                nocolor=args.nocolor,
                                transfer=True)
        val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.threads)
        print("Done")

        print("Create optimizer...", end="", flush=True)

        optimizer_full = torch.optim.Adam(itertools.chain(FGNet.parameters(), dis.parameters()), lr=args.lr, weight_decay=0.0001)  # Discriminator use large learning rate

        optimizer_FGNet = torch.optim.Adam(FGNet.parameters(), lr=args.lr, weight_decay=0.0001)
        optimizer_dis = torch.optim.Adam(dis.parameters(), lr=args.lr, weight_decay=0.0001)  # Discriminator use large learning rate

        # scheduler_FGNet = lr_scheduler.CosineAnnealingLR(optimizer_FGNet, T_max=30, eta_min=1e-5)
        scheduler_lr_decay_full = PolynomialLRDecay(optimizer_full, max_decay_steps=args.epochs, end_learning_rate=0.000001, power=2.0)
        scheduler_lr_decay_FGNet = PolynomialLRDecay(optimizer_FGNet, max_decay_steps=args.epochs, end_learning_rate=0.000001, power=2.0)
        scheduler_lr_decay_dis = PolynomialLRDecay(optimizer_dis, max_decay_steps=args.epochs, end_learning_rate=0.000001, power=2.0)

        print("Done")
        
        # update all para in one optimize is not good

        # create the root folder
        os.makedirs(root_folder, exist_ok=True)
        
        # create the log file
        logs = open(os.path.join(root_folder, "log.txt"), "w")
        logs.write(f"{saved_args}")
        logs.flush()
        logs.write(str(FGNet))
        logs.flush()
        logs.write(str(optimizer_FGNet))
        logs.flush()
        logs.write(str(optimizer_dis))
        logs.flush()

        best_iou = 0.0
        # iterate over epochs
        for epoch in range(args.epochs):

            #######
            # training
            FGNet.train()
            dis.train()

            train_loss = 0
            adv_losses = 0
            val_losses = 0
            iouf = 0
            cm = np.zeros((N_CLASSES, N_CLASSES))
            t = tqdm(zip(train_loader,train_trans_loader), ncols=100, desc="Epoch {}".format(epoch))

            for (pts, features, seg ),(pts_trans, features_trans, seg_trans) in t:
                
                features = features.cuda() # n*3
                pts = pts.cuda()  # n*3
                seg = seg.cuda()

                features_trans = features_trans.cuda() # n*3
                pts_trans = pts_trans.cuda()  # n*3
                seg_trans = seg_trans.cuda()              
            
                # ---------------------
                #  Step1
                # --------------------
                optimizer_full.zero_grad()

                point_features_trans, global_feature_trans = FGNet(features_trans, pts_trans)
                outputs_trans_1, outputs_trans_2 = dis(features_trans, point_features_trans)

                seg_loss_trans_1 = F.cross_entropy(outputs_trans_1.view(-1, N_CLASSES), seg_trans.view(-1))
                seg_loss_trans_2 = F.cross_entropy(outputs_trans_2.view(-1, N_CLASSES), seg_trans.view(-1))

                loss = seg_loss_trans_1+seg_loss_trans_2

                loss.backward()

                optimizer_full.step()


                # ---------------------
                #  Step2
                # --------------------
                optimizer_dis.zero_grad()

                point_features_trans, global_feature_trans = FGNet(features_trans, pts_trans)
                outputs_trans_1, outputs_trans_2 = dis(features_trans, point_features_trans)


                point_features, global_feature = FGNet(features, pts)
                outputs_1, outputs_2 = dis(features, point_features)

                seg_loss_trans_1 = F.cross_entropy(outputs_trans_1.view(-1, N_CLASSES), seg_trans.view(-1))
                seg_loss_trans_2 = F.cross_entropy(outputs_trans_2.view(-1, N_CLASSES), seg_trans.view(-1))

                loss_adv = discrepancy(outputs_1, outputs_2)
                adv_loss = F.l1_loss(outputs_trans_1, outputs_trans_2)

                loss_trans = seg_loss_trans_1+seg_loss_trans_2-loss_adv-adv_loss # 

                loss_trans.backward()

                optimizer_dis.step()


                # ---------------------
                # Step3
                # --------------------

                optimizer_FGNet.zero_grad()

                point_features, global_feature = FGNet(features, pts)
                outputs_1, outputs_2 = dis(features, point_features)
                loss_adv = discrepancy(outputs_1, outputs_2)
                
                point_features_trans,global_feature_trans = FGNet(features_trans, pts_trans)
                outputs_trans_1, outputs_trans_2 = dis(features_trans, point_features_trans)

                seg_loss_trans_1 = F.cross_entropy(outputs_trans_1.view(-1, N_CLASSES), seg_trans.view(-1))
                seg_loss_trans_2 = F.cross_entropy(outputs_trans_2.view(-1, N_CLASSES), seg_trans.view(-1))
                adv_loss = F.l1_loss(outputs_trans_1, outputs_trans_2)

                loss = seg_loss_trans_1 + seg_loss_trans_2 + args.beta*(adv_loss+loss_adv)  #
                loss.backward()
                optimizer_FGNet.step()


                adv_losses += adv_loss.detach().cpu().item()
                # ---------------------
                #  IOU and Loss
                # --------------------
                train_loss += (seg_loss_trans_1 + seg_loss_trans_2).detach().cpu().item()

                outputs = torch.add(outputs_1, outputs_2)/2

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
                target_np = seg.cpu().numpy().copy()   # (16, 8192)

                cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
                cm += cm_

                oa = f"{metrics.stats_overall_accuracy(cm):.5f}"
                aa = f"{metrics.stats_accuracy_per_class(cm)[0]:.5f}"
                iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"

                t.set_postfix(OA=wblue(oa), AA=wblue(aa), IOU=wblue(iou), Train_LOSS=wblue(f"{train_loss/cm.sum():.4e}"), Adv_LOSS=wblue(f"{adv_losses/cm.sum():.4e}"))
    
                 # , Adv_LOSS=wblue(f"{adv_losses/cm.sum():.4e}")

            # write the logs
            logs.write(f"{epoch} {oa} {aa} {iou} {train_loss} "+ " train"+"\n" )
            logs.flush()

            #draw_features(point_features_trans.cpu().detach().numpy().reshape(-1, point_features_trans.shape[2])[0:1000], seg_trans.cpu().detach().numpy().reshape(-1)[0:1000], title =str(epoch))

            if epoch:

                FGNet.eval()
                dis.eval()

                with torch.no_grad(): 
            
                    cm = np.zeros((N_CLASSES, N_CLASSES))
                    t = tqdm(val_loader, ncols=100, desc="Epoch {}".format(epoch))
                    
                    for pts, features, seg in t:

                        features = features.cuda()
                        pts = pts.cuda()
                        seg = seg.cuda()

                        point_features, global_feature = FGNet(features, pts)
                        outputs_1, outputs_2 = dis(features, point_features)

                        outputs = torch.add(outputs_1, outputs_2)/2
                        output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
                        target_np = seg.cpu().numpy().copy()   # (16, 8192)

                        
                        cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
                        cm += cm_

                        oa = f"{metrics.stats_overall_accuracy(cm):.5f}"
                        aa = f"{metrics.stats_accuracy_per_class(cm)[0]:.5f}"
                        iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"

                        iouf = metrics.stats_iou_per_class(cm)[0]          

                        seg_loss_1 = F.cross_entropy(outputs_1.view(-1, N_CLASSES), seg.view(-1))
                        seg_loss_2 = F.cross_entropy(outputs_2.view(-1, N_CLASSES), seg.view(-1))

                        val_losses += (seg_loss_1+seg_loss_2).detach().cpu().item()

                        t.set_postfix(OA=wblue(oa), AA=wblue(aa), IOU=wblue(iou), LOSS=wblue(f"{val_losses/cm.sum():.4e}"))

                    if iouf>best_iou:
                        best_iou = iouf
                        # save the model
                        print("when iou equals ",iou,"save at",os.path.join(root_folder, "state_dict.pth"))
                        torch.save(FGNet.state_dict(), os.path.join(root_folder, "FGNet_state_dict.pth"))
                        torch.save(dis.state_dict(), os.path.join(root_folder, "dis_state_dict.pth"))

                        
                    logs.write(f"{epoch} {oa} {aa} {iou} {val_losses}\n")
                    logs.flush()

        #scheduler_FGNet.step()
        scheduler_lr_decay_full.step()
        scheduler_lr_decay_FGNet.step()
        scheduler_lr_decay_dis.step()

        logs.close()

    ##### TEST
    else:
        # semGen.eval()
        # logs.write("test")
        # logs.flush()
        start = time.time()
        print("start time")
        print(start)
        FGNet.eval()
        dis.eval()

        for filename in filelist_test:
            ds = PartDatasetTest(filename, args.rootdir,
                            block_size=args.block_size,
                            npoints= args.npoints,
                            test_step=args.test_step,
                            nocolor=args.nocolor
                            )
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.threads
                                            )

            xyzrgb = ds.xyzrgb[:,:3]
            scores = np.zeros((xyzrgb.shape[0], N_CLASSES))
            with torch.no_grad():
                t = tqdm(loader, ncols=100)
                for pts, features, indices in t:
                    
                    features = features.cuda()
                    pts = pts.cuda()

                    point_features,_ = FGNet(features, pts)
                    outputs_1, outputs_2 = dis(features, point_features)

                    outputs = torch.add(outputs_1, outputs_2)/2
                    outputs_np = outputs.cpu().numpy().reshape((-1, N_CLASSES))
                    
                    scores[indices.cpu().numpy().ravel()] += outputs_np
                    #print(scores[indices[0][0]])
            
            mask = np.logical_not(scores.sum(1)==0)
            #print(mask)
            scores = scores[mask]
            
            pts_src = xyzrgb[mask]

            # create the scores for all points
            scores = nearest_correspondance(pts_src.astype(np.float32), xyzrgb.astype(np.float32), scores, K=1)

            # compute softmax
            scores = scores - scores.max(axis=1)[:,None]
            scores = np.exp(scores) / np.exp(scores).sum(1)[:,None]
            scores = np.nan_to_num(scores)

            os.makedirs(os.path.join(args.savedir, "results"), exist_ok=True)

            # saving labels
            save_fname = os.path.join(args.savedir, "results", filename.replace(".npy",".labels"))
            scores = scores.argmax(1)
            np.savetxt(save_fname,scores,fmt='%d')

            if args.savepts:
                save_fname = os.path.join(args.savedir, "results", f"{filename}_pts.txt")
                xyzrgb = np.concatenate([xyzrgb, np.expand_dims(scores,1)], axis=1)
                np.savetxt(save_fname,xyzrgb,fmt=['%.4f','%.4f','%.4f','%d'])
            end = time.time()
            print(end - start)
            # break

if __name__ == '__main__':
    main()
    print('{}-Done.'.format(datetime.now()))
