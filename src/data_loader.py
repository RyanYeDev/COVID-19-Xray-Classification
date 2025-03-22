import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
# 数据集路径
train_dir = "../data/xray_dataset_covid19/train"
val_dir   = "../data/xray_dataset_covid19/val"
test_dir  = "../data/xray_dataset_covid19/test"

# 数据预处理
common_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载数据集
train_dataset = ImageFolder(train_dir, common_transform)
val_dataset = ImageFolder(val_dir, common_transform)
test_dataset = ImageFolder(test_dir, common_transform)

# # 类别计数
# targets = train_dataset.targets
# class_counts = np.bincount(targets)
# class_weights = 1.0 / class_counts
# sample_weights = [class_weights[t] for t in targets]
# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# 数据加载器
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader   = DataLoader(val_dataset, batch_size, shuffle=False)
test_dataloader  = DataLoader(test_dataset, batch_size, shuffle=False)

# for images, labels in train_dataloader:
#     print(images.shape)