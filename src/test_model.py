# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.models.mnist_cnn import MNIST_CNN
from src.utils.data_loader import get_data_loaders
from sklearn.metrics import confusion_matrix
import os

def test_model(model_path, optimizer_name=None):
    """Test model and return accuracy, predictions and actual labels"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = MNIST_CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 获取测试数据
    _, test_loader = get_data_loaders(batch_size=1000)
    
    # 测试模型
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")
    
    # 计算混淆矩阵
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    # 如果指定了优化器名称，可视化混淆矩阵
    if optimizer_name:
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {optimizer_name} Optimizer')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    return model, accuracy, all_preds, all_labels, conf_mat

if __name__ == "__main__":
    # 获取当前文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 构建模型路径
    saved_models_dir = os.path.join(project_root, "saved_models")
    adam_model_path = os.path.join(saved_models_dir, "mnist_Adam.pt")
    sgd_model_path = os.path.join(saved_models_dir, "mnist_SGD.pt")
    
    # 测试模型
    print("Testing Adam optimizer model:")
    test_model(adam_model_path, "Adam")
    
    print("\nTesting SGD optimizer model:")
    test_model(sgd_model_path, "SGD")