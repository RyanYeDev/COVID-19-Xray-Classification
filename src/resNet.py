import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import os
from sklearn.metrics import accuracy_score

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
])

# 加载数据集
train_dir = r'D:\COVID-19-Xray-Classification\data\xray_dataset_covid19\train'
val_dir = r'D:\COVID-19-Xray-Classification\data\xray_dataset_covid19\val'

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 加载预训练的ResNet18模型
model_resnet = models.resnet18(pretrained=True)

# 替换最后一层，适应二分类任务
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 2)

# 将模型放到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_resnet.to(device)

# 损失函数：交叉熵损失
criterion = nn.CrossEntropyLoss()

# 优化器：Adam优化器
optimizer = optim.Adam(model_resnet.parameters(), lr=0.001)


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    # 训练和验证过程
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()  # 训练模式
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        start_time = time.time()

        for inputs, labels in train_loader:
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
        model.eval()
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        # 计算验证集准确率
        val_accuracy = correct_preds / total_preds * 100

        # 打印结果
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Accuracy: {val_accuracy:.2f}%, "
              f"Time: {epoch_time:.2f}s")

        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_resnet_model.pth')

    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# 训练和评估模型
train_and_evaluate(model_resnet, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)

# 加载最佳模型
model_resnet.load_state_dict(torch.load('best_resnet_model.pth'))
model_resnet.eval()

# 计算最终准确率（如果有测试集的话）
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# test_accuracy = evaluate_model(model_resnet, test_loader, device)
