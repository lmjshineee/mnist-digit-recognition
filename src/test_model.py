# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.models.mnist_cnn import MNIST_CNN
from src.models.mnist_cnn_improved import MNIST_CNN_Improved

from src.utils.data_loader import get_data_loaders
from sklearn.metrics import confusion_matrix
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def test_model(model_path, optimizer_name):
    """
    测试标准模型并返回准确率和预测结果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 只使用标准 MNIST_CNN 模型
    model = MNIST_CNN().to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 创建测试数据集并加载
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    
    # 数据预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载测试数据集
    test_dataset = datasets.MNIST(data_dir, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 测试模型
    correct = 0
    total = 0
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集预测结果和真实标签
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"{optimizer_name} 模型测试准确率：{accuracy:.2f}%")
    
    # 创建混淆矩阵
    conf_mat = confusion_matrix(all_true_labels, all_predictions)
    
    return model, accuracy, all_predictions, all_true_labels, conf_mat


if __name__ == "__main__":
    # 获取当前文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 构建模型路径
    saved_models_dir = os.path.join(project_root, "saved_models")
    adam_model_path = os.path.join(saved_models_dir, "mnist_Adam.pt")
    sgd_model_path = os.path.join(saved_models_dir, "mnist_SGD.pt")
    improved_model_path = os.path.join(saved_models_dir, "mnist_cnn_mproved.pt")
    
    # 测试模型
    print("Testing Adam optimizer model:")
    test_model(adam_model_path, "Adam")
    
    print("\nTesting SGD optimizer model:")
    test_model(sgd_model_path, "SGD")
    
    print("\nTesting Improved model:")
    test_model(improved_model_path, "Improved")