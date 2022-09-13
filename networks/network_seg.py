from convpoint.nn import PtConv
from convpoint.nn.utils import apply_bn
import torch
import torch.nn as nn
import torch.nn.functional as F

################################
## Shapenet
################################

class SegSmall(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3):
        super(SegSmall, self).__init__()

        n_centers = 16

        pl = 48
        # self.cv1 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        # self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(pl, output_channels)

        # self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(2*pl)

        self.bn5d = nn.BatchNorm1d(2*pl)
        self.bn4d = nn.BatchNorm1d(2*pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        # self.bn0d = nn.BatchNorm1d(pl)

        self.drop = nn.Dropout(0.5)

    def forward(self, x, input_pts, return_features=False):


        # x1, pts1 = self.cv1(x, input_pts, 16, 2048)
        # x1 = F.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x, input_pts, 16, 1024)
        x2 = F.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = F.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 64)  # input, points, K,
        x4 = F.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = F.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = F.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = F.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = F.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = F.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = F.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(x2d, pts2, 8, input_pts)
        x1d = F.relu(apply_bn(x1d, self.bn1d))
        # x1d = torch.cat([x1d, x1], dim=2)

        # x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        # x0d = F.relu(apply_bn(x0d, self.bn0d))

        xout = x1d
        xout = xout.view(-1, xout.size(2))
        xout = self.drop(xout)
        xout = self.fcout(xout)  # Linear
        xout = xout.view(x.size(0), -1, xout.size(1))

        if return_features:
            return xout, x0d
        else:
            return xout

################################
## Shapenet
################################

class SegVerySmall(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3):
        super(SegVerySmall, self).__init__()

        n_centers = 16

        pl = 48
        # self.cv1 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        #self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        #self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        # self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(pl, output_channels)

        # self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        #self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(2*pl)

        self.bn5d = nn.BatchNorm1d(2*pl)
        self.bn4d = nn.BatchNorm1d(2*pl)
        #self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        # self.bn0d = nn.BatchNorm1d(pl)

        self.drop = nn.Dropout(0.5)

    def forward(self, x, input_pts, return_features=False):


        # x1, pts1 = self.cv1(x, input_pts, 16, 2048)
        # x1 = F.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x, input_pts, 16, 1024)
        x2 = F.relu(apply_bn(x2, self.bn2))

        # x3, pts3 = self.cv3(x2, pts2, 16, 256)
        # x3 = F.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x2, pts2, 8, 64)
        x4 = F.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = F.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = F.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = F.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = F.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        # x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        # x3d = F.relu(apply_bn(x3d, self.bn3d))
        # x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x4d, pts4, 8, pts2)  # 16384x3072 and 1536x48
        x2d = F.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(x2d, pts2, 8, input_pts)
        x1d = F.relu(apply_bn(x1d, self.bn1d))
        # x1d = torch.cat([x1d, x1], dim=2)

        # x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        # x0d = F.relu(apply_bn(x0d, self.bn0d))

        xout = x1d
        xout = xout.view(-1, xout.size(2))
        xout = self.drop(xout)
        xout = self.fcout(xout)  # Linear
        xout = xout.view(x.size(0), -1, xout.size(1))

        if return_features:
            return xout, x0d
        else:
            return xout

################################
## S3DIS / NPM3D / SEMANTIC3D
################################

class SegBig(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3, args={}):
        super(SegBig, self).__init__()

        n_centers = 16

        pl = 64
        self.cv0 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv1 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(pl + pl, output_channels)

        self.bn0 = nn.BatchNorm1d(pl)
        self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(2*pl)

        self.bn5d = nn.BatchNorm1d(2*pl)
        self.bn4d = nn.BatchNorm1d(2*pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        self.bn0d = nn.BatchNorm1d(pl)

        if "drop" in args:
            print("Model with dropout")
            self.drop = nn.Dropout(args.drop)
        else:
            self.drop = nn.Dropout(0.0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts, return_features=False):

        x0, _ = self.cv0(x, input_pts, 16)
        x0 = self.relu(apply_bn(x0, self.bn0))

        x1, pts1 = self.cv1(x0, input_pts, 16, 2048)
        x1 = self.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x1, pts1, 16, 1024)
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 64)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = self.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = self.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = self.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = self.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = self.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = self.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(x2d, pts2, 8, pts1)
        x1d = self.relu(apply_bn(x1d, self.bn1d))
        x1d = torch.cat([x1d, x1], dim=2)

        x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        x0d = self.relu(apply_bn(x0d, self.bn0d))

        x0d = torch.cat([x0d, x0], dim=2)

        xout = x0d
        xout = self.drop(xout)
        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        if return_features:
            return xout, x0d
        else:
            return xout


class SegBig_FineTunning(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3, args={}):
        super(SegBig_FineTunning, self).__init__()

        n_centers = 16

        pl = 64
        self.cv0 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv1 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        self.bn0 = nn.BatchNorm1d(pl)
        self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(2*pl)


        for p in self.parameters():
            p.requires_grad = False

        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(pl + pl, output_channels)

        self.bn5d = nn.BatchNorm1d(2*pl)
        self.bn4d = nn.BatchNorm1d(2*pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        self.bn0d = nn.BatchNorm1d(pl)

        if "drop" in args:
            print("Model with dropout")
            self.drop = nn.Dropout(args.drop)
        else:
            self.drop = nn.Dropout(0.0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts, return_features=False):

        x0, _ = self.cv0(x, input_pts, 16)
        x0 = self.relu(apply_bn(x0, self.bn0))

        x1, pts1 = self.cv1(x0, input_pts, 16, 2048)
        x1 = self.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x1, pts1, 16, 1024)
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 64)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = self.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = self.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = self.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = self.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = self.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = self.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(x2d, pts2, 8, pts1)
        x1d = self.relu(apply_bn(x1d, self.bn1d))
        x1d = torch.cat([x1d, x1], dim=2)

        x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        x0d = self.relu(apply_bn(x0d, self.bn0d))

        x0d = torch.cat([x0d, x0], dim=2)

        xout = x0d
        xout = self.drop(xout)
        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        if return_features:
            return xout, x0d
        else:
            return xout


class SegSmall_Discriminator(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3):
        super(SegSmall_Discriminator, self).__init__()

        n_centers = 16

        pl = 48
        # self.cv1 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        # self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(pl, output_channels)

        # unsupervised
        self.ccout = nn.Linear(pl, 1)

        # self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(2*pl)

        self.bn5d = nn.BatchNorm1d(2*pl)
        self.bn4d = nn.BatchNorm1d(2*pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        # self.bn0d = nn.BatchNorm1d(pl)

        self.drop = nn.Dropout(0.5)

    def forward(self, x, input_pts, return_features=False):


        # x1, pts1 = self.cv1(x, input_pts, 16, 2048)
        # x1 = F.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x, input_pts, 16, 1024)
        x2 = F.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = F.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 64)
        x4 = F.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = F.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = F.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = F.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = F.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = F.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = F.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(x2d, pts2, 8, input_pts)
        x1d = F.relu(apply_bn(x1d, self.bn1d))
        # x1d = torch.cat([x1d, x1], dim=2)

        # x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        # x0d = F.relu(apply_bn(x0d, self.bn0d))

        xout = x1d
        xout = xout.view(-1, xout.size(2))
        xout = self.drop(xout)
        xout = self.fcout(xout)  # Linear
        xout = xout.view(x.size(0), -1, xout.size(1))

        
        ## unsupervised model output
        cout = x1d
        cout = cout.view(-1, cout.size(2))
        cout = self.drop(cout)
        cout = self.ccout(cout)  # Linear
        cout = cout.view(x.size(0), -1, cout.size(1))

        if return_features:
            return xout, cout, x0d
        else:
            return xout, cout


################################
## S3DIS / NPM3D / SEMANTIC3D
################################

class SegBig_FG(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3, args={}):
        super(SegBig_FG, self).__init__()

        n_centers = 16

        pl = 64
        self.cv0 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv1 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        # self.fcout = nn.Linear(pl + pl, output_channels)

        self.bn0 = nn.BatchNorm1d(pl)
        self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(2*pl)

        self.bn5d = nn.BatchNorm1d(2*pl)
        self.bn4d = nn.BatchNorm1d(2*pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        self.bn0d = nn.BatchNorm1d(pl)

        if "drop" in args:
            print("Model with dropout")
            self.drop = nn.Dropout(args.drop)
        else:
            self.drop = nn.Dropout(0.0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts, return_features=False):

        x0, _ = self.cv0(x, input_pts, 16)
        x0 = self.relu(apply_bn(x0, self.bn0))

        x1, pts1 = self.cv1(x0, input_pts, 16, 2048)
        x1 = self.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x1, pts1, 16, 1024)
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 64)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = self.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = self.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = self.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = self.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = self.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = self.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(x2d, pts2, 8, pts1)
        x1d = self.relu(apply_bn(x1d, self.bn1d))
        x1d = torch.cat([x1d, x1], dim=2)

        x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        x0d = self.relu(apply_bn(x0d, self.bn0d))

        x0d = torch.cat([x0d, x0], dim=2)

        # xout = x0d
        # xout = self.drop(xout)
        # xout = xout.view(-1, xout.size(2))
        # xout = self.fcout(xout)
        # xout = xout.view(x.size(0), -1, xout.size(1))

        return x0d # , x6



class SegBig_Dis(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3, args={}):
        super(SegBig_Dis, self).__init__()

        n_centers = 16
        pl = 64

        self.fcout = nn.Linear(pl + pl, output_channels)
        self.fcout2 = nn.Linear(pl + pl, output_channels)

        # self.linear1 = nn.Linear(pl + pl, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=0.5)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=0.5)
        # self.linear3 = nn.Linear(256, output_channels)

        if "drop" in args:
            print("Model with dropout")
            self.drop = nn.Dropout(args.drop)
        else:
            self.drop = nn.Dropout(0.0)


    def forward(self, x, x0d, return_features=False):

        xout = x0d
        xout = self.drop(xout)
        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        # classification result
        xout2 = x0d
        xout2 = self.drop(xout2)
        xout2 = xout2.view(-1, xout2.size(2))
        xout2 = self.fcout2(xout2)
        xout2 = xout2.view(x.size(0), -1, xout2.size(1))

        # xout3 = F.leaky_relu(self.bn6(self.linear1(xout2)), negative_slope=0.2)
        # xout3 = self.dp1(xout3)
        # xout4 = F.leaky_relu(self.bn7(self.linear2(xout3)), negative_slope=0.2)
        # xout4 = self.dp2(xout4)
        # xout5 = self.linear3(xout4)
        # xout5 = xout5.view(x.size(0), -1, xout.size(1))


        return xout, xout2

class SegBig_PointDAN_Dis(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3, args={}):
        super(SegBig_PointDAN_Dis, self).__init__()

        n_centers = 16
        pl = 64

        self.fcout = nn.Linear(pl + pl+64, output_channels)
        self.fcout2 = nn.Linear(pl + pl+64, output_channels)

        # self.linear1 = nn.Linear(pl + pl, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=0.5)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=0.5)
        # self.linear3 = nn.Linear(256, output_channels)

        if "drop" in args:
            print("Model with dropout")
            self.drop = nn.Dropout(args.drop)
        else:
            self.drop = nn.Dropout(0.0)


    def forward(self, x, x0d, return_features=False):

        xout = x0d
        xout = self.drop(xout)
        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        # classification result
        xout2 = x0d
        xout2 = self.drop(xout2)
        xout2 = xout2.view(-1, xout2.size(2))
        xout2 = self.fcout2(xout2)
        xout2 = xout2.view(x.size(0), -1, xout2.size(1))

        # xout3 = F.leaky_relu(self.bn6(self.linear1(xout2)), negative_slope=0.2)
        # xout3 = self.dp1(xout3)
        # xout4 = F.leaky_relu(self.bn7(self.linear2(xout3)), negative_slope=0.2)
        # xout4 = self.dp2(xout4)
        # xout5 = self.linear3(xout4)
        # xout5 = xout5.view(x.size(0), -1, xout.size(1))


        return xout, xout2


class Domain_Dis(nn.Module):
    def __init__(self, output_channels, dimension=3, args={}):
        super(Domain_Dis, self).__init__()

        pl = 64
        
        self.relu = nn.ReLU(inplace=True)

        self.fcout1 = nn.Linear(2*pl, 1)
        self.bn1 = nn.BatchNorm1d(8192)
        self.fcout2 = nn.Linear(8192, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fcout3 = nn.Linear(2048, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fcout4 = nn.Linear(512, 1)
        self.soft=nn.LogSoftmax(dim=1)

        self.fcout = nn.Linear(pl + pl, output_channels)

        if "drop" in args:
            print("Model with dropout")
            self.drop = nn.Dropout(args.drop)
        else:
            self.drop = nn.Dropout(0.0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x0d, return_features=False):

        xout = x0d
        xout = self.drop(xout)
        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))


        xout1 = x0d
        xout1 = self.fcout1(xout1)
        xout1 = xout1.view(xout1.size(0), -1)
        xout1 = self.drop(xout1)
        xout1 = self.relu(self.bn1(xout1))
        
        xout2 = self.fcout2(xout1)
        xout2 = self.relu(self.bn2(xout2))
        xout2 = self.drop(xout2)

        xout3 = self.fcout3(xout2)
        xout3 = self.relu(self.bn3(xout3))
        xout3 = self.drop(xout3)
        
        
        xout4 = self.fcout4(xout3)
        xout4 = xout4.view(x.size(0), -1)
        xout4 = self.soft(xout4)

        return xout, xout4

class SegSmall_Features_Generator(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3):
        super(SegSmall_Features_Generator, self).__init__()

        n_centers = 16

        pl = 48
        # self.cv1 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        # self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        # self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(2*pl)


    def forward(self, x, input_pts, return_features=False):

        # x1, pts1 = self.cv1(x, input_pts, 16, 2048)
        # x1 = F.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x, input_pts, 16, 1024)
        x2 = F.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = F.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 64)
        x4 = F.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = F.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = F.relu(apply_bn(x6, self.bn6))

        
        return x6.cuda(), pts6.cuda(), x5.cuda(), pts5.cuda(), x4.cuda(), pts4.cuda(), x3.cuda(), pts3.cuda(), x2.cuda(), pts2.cuda()


class SegSmall_Features_Discriminotor(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3):
        super(SegSmall_Features_Discriminotor, self).__init__()

        n_centers = 16

        pl = 48

        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        # self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(pl, output_channels)

        # unsupervised
        self.ccout = nn.Linear(pl, 1)

        self.bn5d = nn.BatchNorm1d(2*pl)
        self.bn4d = nn.BatchNorm1d(2*pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        # self.bn0d = nn.BatchNorm1d(pl)

        self.drop = nn.Dropout(0.5)

    def forward(self, x, input_pts, x6, pts6, x5, pts5, x4, pts4, x3, pts3, x2, pts2, return_features=False):

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = F.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = F.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = F.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = F.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(x2d, pts2, 8, input_pts)
        x1d = F.relu(apply_bn(x1d, self.bn1d))
        # x1d = torch.cat([x1d, x1], dim=2)

        # x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        # x0d = F.relu(apply_bn(x0d, self.bn0d))

        xout = x1d
        xout = xout.view(-1, xout.size(2))
        xout = self.drop(xout)
        xout = self.fcout(xout)  # Linear
        xout = xout.view(x.size(0), -1, xout.size(1))

        
        ## unsupervised model output
        cout = x1d
        cout = cout.view(-1, cout.size(2))
        cout = self.drop(cout)
        cout = self.ccout(cout)  # Linear
        cout = cout.view(x.size(0), -1, cout.size(1))

        if return_features:
            return xout, cout, x0d
        else:
            return xout, cout


################################
## Shapenet
################################

class SegSmall_FineTuning(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3):
        super(SegSmall_FineTuning, self).__init__()

        n_centers = 16

        pl = 48
        # self.cv1 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        # self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        # self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)

        for p in self.parameters():
            p.requires_grad = False

        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.bn3 = nn.BatchNorm1d(pl) 
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.bn6 = nn.BatchNorm1d(2*pl) 



        
        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.bn5d = nn.BatchNorm1d(2*pl)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.bn4d = nn.BatchNorm1d(2*pl)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.bn3d = nn.BatchNorm1d(pl)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.bn2d = nn.BatchNorm1d(pl)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.bn1d = nn.BatchNorm1d(pl)
        self.drop = nn.Dropout(0.5)
        self.fcout = nn.Linear(pl, output_channels)

        # self.bn0d = nn.BatchNorm1d(pl)


    def forward(self, x, input_pts, return_features=False):


        # x1, pts1 = self.cv1(x, input_pts, 16, 2048)
        # x1 = F.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x, input_pts, 16, 1024)
        x2 = F.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = F.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 64)
        x4 = F.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = F.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = F.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = F.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = F.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = F.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = F.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(x2d, pts2, 8, input_pts)
        x1d = F.relu(apply_bn(x1d, self.bn1d))
        # x1d = torch.cat([x1d, x1], dim=2)

        # x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        # x0d = F.relu(apply_bn(x0d, self.bn0d))

        xout = x1d
        xout = xout.view(-1, xout.size(2))
        xout = self.drop(xout)
        xout = self.fcout(xout)  # Linear
        xout = xout.view(x.size(0), -1, xout.size(1))

        if return_features:
            return xout, x0d
        else:
            return xout



################################
## Shapenet
################################

class SegSmallMMD(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3):
        super(SegSmallMMD, self).__init__()

        n_centers = 16

        pl = 48
        # self.cv1 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        # self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(pl, output_channels)

        # self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(2*pl)

        self.bn5d = nn.BatchNorm1d(2*pl)
        self.bn4d = nn.BatchNorm1d(2*pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        # self.bn0d = nn.BatchNorm1d(pl)

        self.drop = nn.Dropout(0.5)

    def forward(self, x, input_pts, return_features=False):


        # x1, pts1 = self.cv1(x, input_pts, 16, 2048)
        # x1 = F.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x, input_pts, 16, 1024)
        x2 = F.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = F.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 64)  # input, points, K,
        x4 = F.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = F.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = F.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = F.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = F.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = F.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = F.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(x2d, pts2, 8, input_pts)
        x1d = F.relu(apply_bn(x1d, self.bn1d))
        # x1d = torch.cat([x1d, x1], dim=2)

        # x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        # x0d = F.relu(apply_bn(x0d, self.bn0d))

        xout = x1d
        xout = xout.view(-1, xout.size(2))
        xout = self.drop(xout)
        xout = self.fcout(xout)  # Linear
        xout = xout.view(x.size(0), -1, xout.size(1))

        if return_features:
            return xout, x1d
        else:
            return xout


################################
## S3DIS / NPM3D / SEMANTIC3D
################################

class SegBigMMD(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3, args={}):
        super(SegBigMMD, self).__init__()

        n_centers = 16

        pl = 64
        self.cv0 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv1 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(pl + pl, output_channels)

        self.bn0 = nn.BatchNorm1d(pl)
        self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(2*pl)

        self.bn5d = nn.BatchNorm1d(2*pl)
        self.bn4d = nn.BatchNorm1d(2*pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        self.bn0d = nn.BatchNorm1d(pl)

        if "drop" in args:
            print("Model with dropout")
            self.drop = nn.Dropout(args.drop)
        else:
            self.drop = nn.Dropout(0.0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts, return_features=False):

        x0, _ = self.cv0(x, input_pts, 16)
        x0 = self.relu(apply_bn(x0, self.bn0))

        x1, pts1 = self.cv1(x0, input_pts, 16, 2048)
        x1 = self.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x1, pts1, 16, 1024)
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 64)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = self.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = self.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = self.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = self.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = self.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = self.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(x2d, pts2, 8, pts1)
        x1d = self.relu(apply_bn(x1d, self.bn1d))
        x1d = torch.cat([x1d, x1], dim=2)

        x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        x0d = self.relu(apply_bn(x0d, self.bn0d))

        x0d = torch.cat([x0d, x0], dim=2)

        xout = x0d
        xout = self.drop(xout)
        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        if return_features:
            return xout, x0d
        else:
            return xout

