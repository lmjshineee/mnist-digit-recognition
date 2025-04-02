import torch
import os
import urllib.request
import gzip
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert image to PIL format
        image = image.reshape(28, 28).astype(np.uint8)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def download_mnist_files():
    """Manually download MNIST dataset files"""
    # 修改路径，使 data 文件夹位于 src 文件夹的同级目录
    utils_dir = os.path.dirname(__file__)  # utils 目录
    src_dir = os.path.dirname(utils_dir)  # src 目录
    project_root = os.path.dirname(src_dir)  # 项目根目录
    data_dir = os.path.join(project_root, "data", "MNIST", "raw")
    os.makedirs(data_dir, exist_ok=True)
    
    # Define alternative mirror download links
    base_urls = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/"
    ]
    
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    
    # Try downloading from different mirrors
    for file_name in files.values():
        file_path = os.path.join(data_dir, file_name)
        
        # Skip download if file already exists
        if os.path.exists(file_path):
            print(f"File already exists: {file_path}")
            continue
            
        success = False
        for base_url in base_urls:
            url = base_url + file_name
            try:
                print(f"Trying to download {file_name} from {url}...")
                urllib.request.urlretrieve(url, file_path)
                success = True
                print(f"Successfully downloaded to {file_path}")
                break
            except Exception as e:
                print(f"Download failed from {url}: {e}")
                
        if not success:
            print(f"Could not download file {file_name}. Please download manually and place in {data_dir}.")
            print(f"Possible download links:")
            for base_url in base_urls:
                print(f"- {base_url + file_name}")
            return False
    
    return True

def load_mnist_data():
    """Load MNIST data from local files"""
    # 修改路径，使 data 文件夹位于 src 文件夹的同级目录
    utils_dir = os.path.dirname(__file__)  # utils 目录
    src_dir = os.path.dirname(utils_dir)  # src 目录
    project_root = os.path.dirname(src_dir)  # 项目根目录
    data_dir = os.path.join(project_root, "data", "MNIST", "raw")
    
    # Ensure directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if files are already downloaded
    files = {
        "train_images": os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        "train_labels": os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
        "test_images": os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        "test_labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    }
    
    for file_path in files.values():
        if not os.path.exists(file_path):
            success = download_mnist_files()
            if not success:
                raise RuntimeError("Failed to download MNIST dataset files. Check your network connection or download manually.")
            break
    
    # Load training images
    with gzip.open(files["train_images"], 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    
    # Load training labels
    with gzip.open(files["train_labels"], 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    # Load test images
    with gzip.open(files["test_images"], 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    
    # Load test labels
    with gzip.open(files["test_labels"], 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    return (train_images, train_labels), (test_images, test_labels)

def get_data_loaders(batch_size=64):
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        # First try using torchvision's built-in download
        # 修改路径，使 data 文件夹位于 src 文件夹的同级目录
        utils_dir = os.path.dirname(__file__)  # utils 目录
        src_dir = os.path.dirname(utils_dir)  # src 目录
        project_root = os.path.dirname(src_dir)  # 项目根目录
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
        
        print("Successfully downloaded MNIST dataset using torchvision")
        
    except Exception as e:
        print(f"Failed to download MNIST dataset using torchvision: {e}")
        print("Trying manual download and loading...")
        
        # Load data manually
        (train_images, train_labels), (test_images, test_labels) = load_mnist_data()
        
        # Create datasets
        train_dataset = MNISTDataset(train_images, train_labels, transform=transform)
        test_dataset = MNISTDataset(test_images, test_labels, transform=transform)
        
        print("Successfully loaded MNIST dataset manually")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader