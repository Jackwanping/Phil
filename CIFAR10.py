import torch
import torchvision
import torch.nn as nn
import time
# 准备数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10(root='../Data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../Data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
device = torch.device("cuda")
# 获取数据长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(train_data_size)
print(test_data_size)

# 创建dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x
CNN = NN()
NN.to(device)
# 设置损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 设置优化器
learn_rate = 1e-2
optimizer = torch.optim.SGD(NN.parameters(), lr=learn_rate)

# 设置训练、测试次数，训练的轮数
total_train_step = 0
total_test_step = 0
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs_train")
# 开始训练
CNN.train()
start_time = time.time()
for i in range(epoch):
    print("----训练第{}轮开始".format(i+1))
    for data in train_dataloader:
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)
        output = CNN(images)

        # 设置loss
        loss = loss_fn(output, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练100步所需要的时间：{}".format(end_time-start_time))
            print("训练次数:{},Loss值的大小:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 开始测试
    CNN.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            output = NN(images)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("测试集上的loss{}".format(total_test_loss))
    print("测试的正确率是:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", loss.item(), total_test_loss)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(NN, "CNN_{}.pth".format(i))
    print("模型已经保存")

writer.close()










if __name__ == '__main__':
    CNN = NN()
    input = torch.ones((64, 3, 32, 32))
    output = CNN(input)
    print(output.shape)


