import os
import sys
import torch

# 添加项目根目录到 Python 路径
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
sys.path.append(project_root)

from src.models.train_model import train_model
from src.test_model import test_model

def main():
    # 训练模型并保存
    train_model()
    
    # 构建 saved_models 的路径
    saved_models_dir = os.path.join(project_root, "saved_models")
    
    # 测试模型
    adam_path = os.path.join(saved_models_dir, "mnist_Adam.pt")
    sgd_path = os.path.join(saved_models_dir, "mnist_SGD.pt")
    
    print(f"测试 Adam 模型：{adam_path}")
    test_model(adam_path, "Adam")
    
    print(f"测试 SGD 模型：{sgd_path}")
    test_model(sgd_path, "SGD")

if __name__ == "__main__":
    main()