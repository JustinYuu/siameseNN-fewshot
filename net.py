import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SiameseNN(nn.Module):
    def __init__(self):
        super(SiameseNN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10),   # 96*96*64
            nn.ReLU(True),
            nn.MaxPool2d(2),    # 48*48*64
            nn.Conv2d(64, 128, kernel_size=7),  # 42*42*128
            nn.ReLU(True),
            nn.MaxPool2d(2),    # 21*21*128
            nn.Conv2d(128, 128, kernel_size=4),     # 18*18*128
            nn.ReLU(True),
            nn.MaxPool2d(2),    # 9*9*128
            nn.Conv2d(128, 256, kernel_size=4),     # 6*6*256
            nn.ReLU(True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.Sigmoid()
        )
        self.fc2 = nn.Linear(4096, 1)   # BCEWithLogitsLoss()

    def forward(self, x1, x2):
        out1 = self.feature(x1)
        out2 = self.feature(x2)
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out1 = self.fc1(out1)
        out2 = self.fc1(out2)
        out = torch.abs(out1-out2)
        out = self.fc2(out)
        return out

