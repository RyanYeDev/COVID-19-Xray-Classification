import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
# 数据集路径
train_dir = "../data/xray_dataset_covid19/train"
test_dir = "../data/xray_dataset_covid19/test"

# 数据预处理
train_transform = torchvision.transforms.Compose([
    transforms.Resize((224, 224)),          # 调整图像大小
    transforms.RandomHorizontalFlip(),      # 数据增强：随机水平翻转
    transforms.RandomRotation(10),          # 数据增强：随机旋转
    transforms.ToTensor(),                  # 转换为张量
    # transforms.Normalize(mean=[0.485], std=[0.229])  # 归一化（单通道）
])
test_transform = torchvision.transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485], std=[0.229])
])

# 加载数据集
train_dataset = ImageFolder(train_dir, transform=train_transform)
test_dataset = ImageFolder(test_dir, transform=test_transform)

# 数据集长度
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)

train_dataloader = DataLoader(train_dataset, 1, True)
test_dataloader = DataLoader(test_dataset, 1, False)