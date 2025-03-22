import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from model import *
from data_loader import *

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CovidCNN(num_classes=2).to(device)  # 二分类

# 损失函数 & 优化器
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard 日志记录
writer = SummaryWriter(f"logs/exp_{int(time.time())}")

# 超参数
num_epochs = 10

# 训练过程
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # 训练循环
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / len(train_dataloader)
    train_accuracy = 100 * train_correct / train_total

    # ======= 验证阶段 =======
    val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)

    # ======= 测试阶段 =======
    test_loss, test_accuracy = evaluate(model, test_dataloader, criterion, device)

    # ======= 输出与记录 =======
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
    print(f"  Val   Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    print(f"  Test  Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    # TensorBoard 记录
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_accuracy, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_accuracy, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", test_accuracy, epoch)

    # ======= 保存模型 =======
    model_save_path = f"../models/model_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"  Model saved to {model_save_path}")

writer.close()
