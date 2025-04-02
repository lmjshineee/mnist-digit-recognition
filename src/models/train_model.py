import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from src.models.mnist_cnn import MNIST_CNN
from src.utils.data_loader import get_data_loaders

def train_model(epochs=5, batch_size=64, learning_rate=0.001):
    # 修改保存目录路径为 src 同级的 saved_models 目录
    models_dir = os.path.dirname(__file__)  # models 目录
    src_dir = os.path.dirname(models_dir)  # src 目录
    project_root = os.path.dirname(src_dir)  # 项目根目录
    save_dir = os.path.join(project_root, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, _ = get_data_loaders(batch_size)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train with Adam optimizer
    model_adam = MNIST_CNN().to(device)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training with Adam optimizer...")
    train_optimizer(model_adam, train_loader, optimizer_adam, criterion, epochs, device)
    
    adam_path = os.path.join(save_dir, "mnist_Adam.pt")
    torch.save(model_adam.state_dict(), adam_path)
    print(f"Adam model saved to '{adam_path}'")
    
    # Train with SGD optimizer
    model_sgd = MNIST_CNN().to(device)
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=learning_rate, momentum=0.9)
    
    print("Starting training with SGD optimizer...")
    train_optimizer(model_sgd, train_loader, optimizer_sgd, criterion, epochs, device)
    
    sgd_path = os.path.join(save_dir, "mnist_SGD.pt")
    torch.save(model_sgd.state_dict(), sgd_path)
    print(f"SGD model saved to '{sgd_path}'")

def train_optimizer(model, train_loader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/100:.4f}')
                running_loss = 0.0
                
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1} completed, time: {epoch_time:.2f} seconds')

if __name__ == "__main__":
    train_model(epochs=5, batch_size=64, learning_rate=0.001)