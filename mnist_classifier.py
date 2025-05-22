#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MNIST手写数字识别 - 深度学习经典案例
使用PyTorch框架实现卷积神经网络(CNN)识别手写数字
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

# 设备配置 - 如果有GPU则使用GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数设置
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# 数据预处理和加载
# transforms.ToTensor()将图像转换为PyTorch张量,并归一化到[0,1]
# transforms.Normalize将数据标准化到均值为0.1307,标准差为0.3081
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载训练集 - 首次运行会自动下载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 查看一些样本数据
def show_samples(dataloader, num_samples=5):
    """显示数据集中的样本图像"""
    examples = iter(dataloader)
    example_data, example_targets = next(examples)
    
    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(example_data[i][0], cmap='gray')
        plt.title(f"标签: {example_targets[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_samples.png')  # 保存样本图像
    plt.close()
    print(f"样本图像已保存为 mnist_samples.png")

# 定义卷积神经网络模型
class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        # 第一个卷积层 - 1个输入通道,32个输出通道,3x3卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 第二个卷积层 - 32个输入通道,64个输出通道,3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 最大池化层 - 2x2窗口
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Dropout层 - 防止过拟合
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10个输出对应10个数字类别

    def forward(self, x):
        # 第一个卷积+激活+池化
        x = F.relu(self.conv1(x))  # -> 32x28x28
        x = self.pool(x)  # -> 32x14x14
        
        # 第二个卷积+激活+池化
        x = F.relu(self.conv2(x))  # -> 64x14x14
        x = self.pool(x)  # -> 64x7x7
        
        x = self.dropout1(x)
        # 展平张量
        x = torch.flatten(x, 1)  # -> 64*7*7
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # 输出概率分布
        return F.log_softmax(x, dim=1)

# 初始化模型
model = MNISTConvNet().to(device)
print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()  # 设置为训练模式
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和标签移动到GPU(如果可用)
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 累计损失
        train_loss += loss.item()
        
        # 计算准确率
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 打印训练进度
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    train_loss /= len(train_loader)
    accuracy = 100. * correct / total
    print(f'训练集: 平均损失: {train_loss:.4f}, 准确率: {correct}/{total} ({accuracy:.2f}%)')
    return train_loss, accuracy

# 测试函数
def test(model, device, test_loader):
    model.eval()  # 设置为评估模式
    test_loss = 0
    correct = 0
    total = 0
    
    # 不计算梯度
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            test_loss += criterion(output, target).item()
            
            # 计算准确率
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{total} ({accuracy:.2f}%)')
    return test_loss, accuracy

# 可视化一些预测结果
def visualize_predictions(model, device, test_loader, num_samples=5):
    model.eval()
    examples = iter(test_loader)
    example_data, example_targets = next(examples)
    example_data, example_targets = example_data.to(device), example_targets.to(device)
    
    with torch.no_grad():
        output = model(example_data)
        _, predicted = output.max(1)
    
    plt.figure(figsize=(12, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(example_data[i][0].cpu().numpy(), cmap='gray')
        plt.title(f"真实: {example_targets[i]}\n预测: {predicted[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    plt.close()
    print(f"预测结果已保存为 mnist_predictions.png")

# 可视化学习曲线
def plot_learning_curves(train_losses, train_accs, test_losses, test_accs):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, test_losses, 'r-', label='测试损失')
    plt.title('损失曲线')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='训练准确率')
    plt.plot(epochs, test_accs, 'r-', label='测试准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()
    print(f"学习曲线已保存为 learning_curves.png")

def main():
    # 显示一些样本数据
    print("显示MNIST样本数据...")
    show_samples(train_loader)
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # 训练和测试模型
    print("\n开始模型训练...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    
    # 可视化预测结果
    print("\n可视化模型预测结果...")
    visualize_predictions(model, device, test_loader)
    
    # 绘制学习曲线
    print("\n绘制学习曲线...")
    plot_learning_curves(train_losses, train_accs, test_losses, test_accs)
    
    # 保存模型
    print("\n保存模型...")
    torch.save(model.state_dict(), "mnist_cnn_model.pth")
    print("模型已保存为 mnist_cnn_model.pth")
    
    print("\n程序完成!")

if __name__ == "__main__":
    main()
