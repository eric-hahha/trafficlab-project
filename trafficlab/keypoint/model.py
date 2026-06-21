"""Keypoint + orientation model (Pirazh ICCV 2019), ported to modern PyTorch."""

import torch
import torch.nn as nn
from torchvision import models


class CoarseRegressor(nn.Module):
    def __init__(self, n1=20):
        super().__init__()
        self.N1 = n1
        vgg_layers = list(models.vgg16_bn(weights=None).children())[0]
        children = list(vgg_layers.children())
        self.A1 = nn.Sequential(*children[:7])
        self.A2 = nn.Sequential(*children[7:14])
        self.A3 = nn.Sequential(*children[14:24])
        self.A4 = nn.Sequential(*children[24:34])
        self.A5 = nn.Sequential(*children[34:])
        self.A6 = nn.Sequential(nn.Conv2d(512, 512, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.A6to7 = nn.Sequential(nn.Conv2d(512, n1 + 1, 1), nn.BatchNorm2d(n1 + 1), nn.ReLU())
        self.A3to7 = nn.Sequential(nn.Conv2d(256, n1 + 1, 1), nn.BatchNorm2d(n1 + 1), nn.ReLU())
        self.A4to7 = nn.Sequential(nn.Conv2d(512, n1 + 1, 1), nn.BatchNorm2d(n1 + 1), nn.ReLU())
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.A1(x)
        x = self.A2(x)
        x = self.A3(x)
        res2 = self.A4(x)
        return self.Up(self.Up(self.Up(self.A6to7(self.A6(self.A5(res2)))) + self.A4to7(res2)) + self.A3to7(x))


class FineRegressor(nn.Module):
    def __init__(self, n2=20):
        super().__init__()
        self.N2 = n2
        self.Normalize = nn.Softmax(dim=2)
        self.MaxPool = nn.MaxPool2d(2, 2)
        self.L1 = nn.Sequential(nn.Conv2d(24, 64, 7), nn.BatchNorm2d(64), nn.ReLU())
        self.HR1 = nn.Sequential(
            nn.Conv2d(64, 64, 7), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 5), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 7), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.res1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.L2 = nn.Sequential(
            nn.ConvTranspose2d(64, n2 + 1, 7), nn.BatchNorm2d(n2 + 1), nn.ReLU(),
            nn.Conv2d(n2 + 1, 64, 7), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.L3 = nn.Sequential(
            nn.Conv2d(64, 64, 7), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 5), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.res2 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.L4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 7), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.L5 = nn.Sequential(nn.ConvTranspose2d(64, n2 + 1, 7), nn.BatchNorm2d(n2 + 1), nn.ReLU())
        self.pose_branch1 = nn.Sequential(
            nn.Conv2d(256, 128, 7), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 7), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pose_branch2 = nn.Sequential(nn.Conv2d(64, 32, 7), nn.BatchNorm2d(32), nn.ReLU())
        self.FC = nn.Sequential(nn.Linear(2048, 256), nn.Dropout(0.5), nn.Linear(256, 8))

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(self.res1(x) + self.HR1(x))
        joint = self.L3(x)
        kp = self.L5(self.res2(x) + self.L4(joint))
        B, C, H, W = kp.shape
        kp = self.Normalize(kp.view(B, C, W * H)).view(B, C, H, W)
        pose = self.pose_branch1(joint)
        pose = self.MaxPool(pose)
        pose = self.pose_branch2(pose)
        pose = self.FC(pose.view(-1, 2048))
        return kp, pose


class KeyPointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse_estimator = CoarseRegressor()
        self.refinement = FineRegressor()

    def forward(self, x1, x2):
        coarse_kp = self.coarse_estimator(x1)
        fine_kp, orientation = self.refinement(torch.cat((x2, coarse_kp), dim=1))
        return coarse_kp, fine_kp, orientation
