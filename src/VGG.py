import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import os

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
])

# 加载数据集
train_dir = r'D:\COVID-19-Xray-Classification\data\xray_dataset_covid19\train'
val_dir = r'D:\COVID-19-Xray-Classification\data\xray_dataset_covid19\val'

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 选择设备：GPU（如果可用）或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练的VGG16模型
model_vgg = models.vgg19(pretrained=True)

# 替换最后一层（fc层），适应二分类任务
model_vgg.classifier[6] = nn.Linear(model_vgg.classifier[6].in_features, 2)

# 将模型移动到设备（GPU/CPU）
model_vgg.to(device)

# 损失函数：交叉熵损失
criterion = nn.CrossEntropyLoss()

# 优化器：Adam优化器
optimizer = optim.Adam(model_vgg.parameters(), lr=0.001)

# 训练和评估过程
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()  # 训练模式
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        start_time = time.time()

        for inputs, labels in train_loader:
            # 确保数据也在GPU上
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计损失
            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        # 计算训练集准确率
        train_accuracy = correct_preds / total_preds * 100

        # 验证阶段
        model.eval()  # 验证模式
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                # 确保数据也在GPU上
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        # 计算验证集准确率
        val_accuracy = correct_preds / total_preds * 100

        # 打印结果
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Accuracy: {val_accuracy:.2f}%, "
              f"Time: {epoch_time:.2f}s")

        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_vgg_model.pth')

    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# 训练并评估VGG模型
train_and_evaluate(model_vgg, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)
