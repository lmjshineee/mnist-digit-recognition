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
    ���Ա�׼ģ�Ͳ�����׼ȷ�ʺ�Ԥ����
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ֻʹ�ñ�׼ MNIST_CNN ģ��
    model = MNIST_CNN().to(device)
    
    # ����ģ��Ȩ��
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # �����������ݼ�������
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    
    # ����Ԥ����
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # ���ز������ݼ�
    test_dataset = datasets.MNIST(data_dir, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # ����ģ��
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
            
            # �ռ�Ԥ��������ʵ��ǩ
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"{optimizer_name} ģ�Ͳ���׼ȷ�ʣ�{accuracy:.2f}%")
    
    # ������������
    conf_mat = confusion_matrix(all_true_labels, all_predictions)
    
    return model, accuracy, all_predictions, all_true_labels, conf_mat


if __name__ == "__main__":
    # ��ȡ��ǰ�ļ�·��
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # ����ģ��·��
    saved_models_dir = os.path.join(project_root, "saved_models")
    adam_model_path = os.path.join(saved_models_dir, "mnist_Adam.pt")
    sgd_model_path = os.path.join(saved_models_dir, "mnist_SGD.pt")
    improved_model_path = os.path.join(saved_models_dir, "mnist_cnn_mproved.pt")
    
    # ����ģ��
    print("Testing Adam optimizer model:")
    test_model(adam_model_path, "Adam")
    
    print("\nTesting SGD optimizer model:")
    test_model(sgd_model_path, "SGD")
    
    print("\nTesting Improved model:")
    test_model(improved_model_path, "Improved")