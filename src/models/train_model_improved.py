import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.mnist_cnn_improved import MNIST_CNN_Improved

# ʹ��������ǿ�������ת�ͷ���任�������й�һ��
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ���Լ�ֻ����һ������
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ���� MNIST ���ݼ�
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN_Improved().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# ѧϰ�ʵ�������������֤�� loss �½��������ѧϰ��
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, min_lr=1e-5)

num_epochs = 20
best_val_loss = float('inf')
patience = 5   # ������� 5 �� epoch ��֤�� loss ���½�������ǰֹͣ
trigger_times = 0

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Accuracy = {accuracy:.2f}%')
    
    # ����ѧϰ��
    scheduler.step(val_loss)
    
    # ��¼����ģ�Ͳ������ǰֹͣ����
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), './saved_models/mnist_cnn_improved.pt')
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered!")
            break
