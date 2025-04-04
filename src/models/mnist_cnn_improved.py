import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN_Improved(nn.Module):
    def __init__(self):
        super(MNIST_CNN_Improved, self).__init__()
        # 第一层卷积：输入 1 通道，输出 32 通道，3x3 卷积核，padding 保持尺寸
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 第二层卷积：输入 32 通道，输出 64 通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.25)
        # 全连接层：根据池化后的特征尺寸计算输入节点数（28x28 经过两次 2x2 池化后为 7x7）
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # 尺寸变为 14x14
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 尺寸变为 7x7
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
