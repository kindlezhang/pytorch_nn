import argparse  # 用于解析命令行参数
import torch 
import torch.nn as nn
import torch.optim as optim  # PyTorch中的优化器
from torch.utils.data import DataLoader  # PyTorch中用于加载数据的工具
from torchvision import datasets  # PyTorch中的视觉数据集
import torchvision.transforms as transforms  # PyTorch中的数据变换操作
from tqdm import tqdm  # 用于在循环中显示进度条
import os  # Python中的操作系统相关功能
from model.cnn import simplecnn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# 设备的选择 cpu or gpu

# 对图像做变换
train_transformer = transforms.Compose([
    transforms.Resize([224,224]), # 将数据裁剪为224*224大小
    transforms.ToTensor(), # 把图片转换为 tensor张量 0-1的像素值
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # 标准化
])

test_transformer = transforms.Compose([
    transforms.Resize([224,224]), # 将数据裁剪为224*224大小
    transforms.ToTensor(), # 把图片转换为 tensor张量 0-1的像素值
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # 标准化，每个通道都按0.5均值，0.5标准差的方法标准化，像素变为-1到1
])

# 加载训练集和测试集
trainset = datasets.ImageFolder(root=os.path.join(r"data/COVID_19_Radiography_Dataset","train"), # 拼接路径 找到训练集
                                transform=train_transformer) # 训练集做图像变换

testset = datasets.ImageFolder(root=os.path.join(r"data/COVID_19_Radiography_Dataset","test"),
                               transform=test_transformer)

# 定义训练集的加载器
train_loader = DataLoader(trainset,batch_size=32,num_workers=0,shuffle=True) 
           # trainset传入的训练集，batch 批次训练的图像数量
           # num_workers 数据加载多线程 为0代表不打开  
           # shuffle 为Ture代表打乱加载数据
# 定义测试集的加载器
test_loader = DataLoader(testset,batch_size=32,num_workers=0,shuffle=False)

def train(model, train_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader,desc = f"epoch:{epoch+1}/{num_epochs}", unit = "batch"):
            inputs,labels = inputs.to(device),labels.to(device) # 将数据传到设备上
            optimizer.zero_grad() # 梯度清零
            outputs = model(inputs) # 前向传播
            loss = criterion(outputs,labels) # loss的计算
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            running_loss += loss.item() * inputs.size(0) # 用loss乘批次大小 得到该批次的loss
        epoch_loss = running_loss/len(train_loader.dataset) # 总损失/总数据集大小
        print(f"epoch{epoch+1}/{num_epochs}, Train_loss{epoch_loss:.4f}")

        accuracy = evaluate(model, test_loader, criterion)
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, save_path)
            print("model saved with best acc")

def evaluate(model, test_loader, criterion):
    model.eval() # 指定模型为验证模式
    test_loss = 0.0 
    correct = 0 # 正确样本数量
    total = 0 # 总样本数量
    with torch.no_grad(): # 评估模式下不需要梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device) # 数据送到设备里面
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss = test_loss + loss.item() * inputs.size(0) # 计算批次损失
            _ , predicted = torch.max(outputs, 1) # 返回最大值和索引
            total += labels.size(0) # 计算总样本数量
            correct += (predicted == labels).sum().item() # 正确样本数

    avg_loss = test_loss / len(test_loader.dataset) # 计算平均loss
    accuracy = 100.0 * correct / total # 计算准确率
    print(f"test loss:{avg_loss:.4f}, Accuracy:{accuracy:.2f}%")
    return accuracy

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    num_epochs = 1
    learning_rate = 0.001
    num_class = 4
    save_path = "model_pth/best.pth"
    model = simplecnn(num_class).to(device) # 对模型进行实例化 并进入gpu或者cpu中
    criterion = nn.CrossEntropyLoss() # 指定损失函数为交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr = learning_rate) # 指定优化器为adam，或者其他的比如sgd
    train(model, train_loader, criterion, optimizer, num_epochs) # 使用训练集训练
    evaluate(model, test_loader, criterion) # 使用测试集进行测试
