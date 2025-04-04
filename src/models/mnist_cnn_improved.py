import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN_Improved(nn.Module):
    def __init__(self):
        super(MNIST_CNN_Improved, self).__init__()
        # ��һ���������� 1 ͨ������� 32 ͨ����3x3 ����ˣ�padding ���ֳߴ�
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # �ڶ����������� 32 ͨ������� 64 ͨ��
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # �ػ���
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout ��ֹ�����
        self.dropout = nn.Dropout(0.25)
        # ȫ���Ӳ㣺���ݳػ���������ߴ��������ڵ�����28x28 �������� 2x2 �ػ���Ϊ 7x7��
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # �ߴ��Ϊ 14x14
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # �ߴ��Ϊ 7x7
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # չƽ
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
