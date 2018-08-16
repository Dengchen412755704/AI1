# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:41:55 2018

@author: dengchen
"""
#导入函数模块
#import gzip,struct
#import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.data import TensorDataset,DataLoader
from torchvision import datasets, transforms
import math

#定义一个LeNet神经网络结构
#The network is like:
def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(self.dropout(out))
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])
       
    #NOTE: You don't need to define backward function, Pytorch can
    #automatically cal gradients and update weights.
    
#判定GPU是否存在
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#定义超参数
batch_size = 100
learning_rate = 0.001
kwargs = {'num_workers': 2, 'pin_memory': True}            #DataLoader的参数
        
#参数值初始化
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weigth.data.fill_(1)
        m.bias.data.zero_()
      
            
#定义训练函数  
def train(epoch):
    #调用前向传播
    model.train()
    for batch_idx,(data, target) in enumerate(train_loader):
         data, target = data.to(device), target.to(device)
         data,target = Variable(data), Variable(target)  #定义为Variable类型，能够调用autograd
         #初始化时，要清空梯度
         #initial optimizer,将网络中的所有梯度置0
         optimizer.zero_grad()
         # net work will do forward computation defined in net's [forward]
         output = model(data)
         
         # get predictions from outputs, the highest score's index in a vector 
         predictions = output.max(1, keepdim=True)[1]
         # cal correct predictions num
         correct = predictions.eq(target.view_as(predictions)).sum().item()
         # cal accuracy
         acc = correct / len(target)
         
         # use loss func to cal loss
         loss = loss_func(output, target)#预测的输出和实际值的差值
         # backward will back propagate loss
         loss.backward()
         # this will update all weights use the loss we just back propagate
         optimizer.step()
         
         
         if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{}  ({:.0f}%)], Loss: {:.4f}, Accuracy: {:.2f}'
                  .format(epoch,batch_idx * len(data), len(train_loader.dataset),
                          100.*batch_idx / len(train_loader), loss.data[0], acc))
            

#定义测试函数
def test():
    model.eval()#让模型变为测试模式，主要是保证dropout和BN和训练过程一致。BN是指batch normalization
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data,target = Variable(data), Variable(target) 
        output = model(data)
        #计算总的损失
        test_loss += F.nll_loss(output, target).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

           
#手写体数据导入
#处理数据  torch.utils.data.DataLoader,对图像和标签分别封装成一个Tensor
train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True,
                       download=True,
                       transform=transforms.Compose([transforms.ToTensor()])),
                       batch_size=batch_size,
                       shuffle=True
)
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False,
                       transform=transforms.Compose([transforms.ToTensor()])),
                       batch_size=batch_size,
                       shuffle=False                   
)
        

#define network定义模型
model = resnet32()
# define loss function 定义损失函数 使用交叉熵验证
loss_func = torch.nn.CrossEntropyLoss(size_average=False)
# define optimizer 定义优化算法   直接定义优化器，而不是调用backward
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))


#调用参数初始化方法初始化网络参数
model.apply(_weights_init)#_weights_init(m)

#调用函数执行训练和测试
#训练和评估
# start train
for epoch in range(1, 30):
    print('----------------start train-----------------')
    train(epoch)
    print('----------------end train-----------------')

    print('----------------start test-----------------')
    test()
    print('----------------end test-----------------\n')