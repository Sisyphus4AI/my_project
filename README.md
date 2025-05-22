# MNIST手写数字识别项目

## 项目简介
这是一个基于PyTorch的MNIST手写数字识别深度学习项目，实现了一个卷积神经网络(CNN)来识别手写数字。MNIST是一个包含大量手写数字(0-9)的数据集，是深度学习领域的经典入门案例。

## 项目内容
本项目实现了以下内容：
- 数据加载与预处理
- 构建卷积神经网络模型
- 模型训练与测试
- 模型评估与可视化

## 数据集
MNIST数据集包含：
- 训练集：60,000张手写数字图像及其标签
- 测试集：10,000张手写数字图像及其标签
- 图像尺寸：28x28像素，灰度图像

本项目使用PyTorch的`torchvision.datasets.MNIST`类自动下载并加载数据集，数据会保存在`./data`目录下。

## 环境要求
运行本项目需要安装以下Python库：
```
pytorch >= 1.7.0
torchvision
matplotlib
numpy
```

可以通过Anaconda创建虚拟环境并安装依赖：
```bash
conda create -n mnist_env python=3.8
conda activate mnist_env
conda install pytorch torchvision matplotlib numpy -c pytorch
```

## 文件说明
- `mnist_classifier.py`：主要代码文件，包含完整的MNIST手写数字识别实现

## 运行方法
```bash
python mnist_classifier.py
```

## 输出结果
程序运行后会生成以下文件：
1. `mnist_samples.png` - 显示MNIST数据集的样本图像
2. `mnist_predictions.png` - 显示模型预测结果
3. `learning_curves.png` - 显示训练和测试过程中的损失和准确率曲线
4. `mnist_cnn_model.pth` - 训练好的模型参数文件

## 模型架构
本项目使用的卷积神经网络(CNN)模型结构如下：
1. 卷积层1: 1个输入通道，32个输出通道，3x3卷积核
2. ReLU激活 + 2x2最大池化
3. 卷积层2: 32个输入通道，64个输出通道，3x3卷积核
4. ReLU激活 + 2x2最大池化
5. Dropout层(防止过拟合)
6. 全连接层1: 64*7*7输入特征，128输出特征
7. ReLU激活 + Dropout
8. 全连接层2: 128输入特征，10输出特征(对应10个数字类别)
9. Log Softmax输出

## 预期结果
经过5个Epoch的训练，模型在测试集上的准确率应该达到98%以上。具体表现如下：
- 训练集准确率：应该达到99%以上
- 测试集准确率：应该达到98%以上

## 参考资源
- [MNIST数据集官方网站](http://yann.lecun.com/exdb/mnist/)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch教程 - 训练分类器](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## 问题导向学习
根据问题导向学习方法，这里提供对应的分析：

### 1. 这个模块的作用是什么
`mnist_classifier.py`模块的主要作用是实现一个完整的深度学习工作流程，包括数据加载、数据预处理、模型定义、训练、测试和可视化。

### 2. 这个模块有哪些重要的类和函数
重要的类：
- `MNISTConvNet`: 定义了卷积神经网络模型结构

重要的函数：
- `train()`: 负责模型训练的函数
- `test()`: 负责模型测试和评估的函数
- `show_samples()`: 显示样本数据的函数
- `visualize_predictions()`: 可视化模型预测结果的函数
- `plot_learning_curves()`: 绘制学习曲线的函数

### 3. 这些重要的类和函数在哪里调用的，作用是什么
- `MNISTConvNet`类在主程序中被实例化，用于创建神经网络模型
- `train()`函数在每个训练周期(epoch)中被调用，负责模型参数更新
- `test()`函数在每个训练周期后被调用，评估模型在测试集上的性能
- 可视化函数在相应的程序阶段被调用，生成图像帮助理解数据和结果

### 4. 常用的代码块
数据加载和预处理：
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, 
                              download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
```

训练循环：
```python
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
    test_loss, test_acc = test(model, device, test_loader)
```

### 5. 数据输入输出的格式
- 输入格式：28x28像素的灰度图像，经过预处理转换为形状为[1, 28, 28]的PyTorch张量
- 输出格式：长度为10的一维张量，表示模型对0-9每个数字的预测概率

## 进阶学习方向
掌握这个项目后，你可以尝试以下进阶内容：
1. 修改网络架构，尝试不同的层数和参数配置
2. 尝试不同的优化器和学习率调度策略
3. 实现数据增强技术提高模型泛化能力
4. 尝试将模型部署到实际应用中
5. 尝试其他经典数据集，如CIFAR-10、Fashion-MNIST等