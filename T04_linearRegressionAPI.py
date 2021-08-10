# -*- coding:utf-8 -*-
"""
author: 15025
time: 10.08.2021   18:41:55
software: PyCharm

Description:
    uss API of Pytorch to realise linear regression
    torch.nn.Linear为预定义好的全连接层，
    1.定义模型
    2.优化器
        torch.optim.SGD(参数，学习率)
        torch.optim.Adam(参数，学习率)
    优化器的使用方法：
        optimizer = optim.SGD(model.parameters(), lr=1e-3)  # 实例化
        optimizer.zero_grad()   # 梯度置为0
        loss.backward() # 计算梯度
        optimizer.step()    # 更新参数的值
    3.损失函数
        torch.nn.MSELoss() 均方误差
        torch.nn.CrossEntropyLoss()  交叉熵损失
    损失函数使用方法
        model = lr()    # 实例化模型
        criterion = nn.MSELoss()    # 实例化损失函数
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        for i in range(100):
            y_predict = model(x_true)   # 向前计算预测值
            loss = criterion(y_true, y_predict) # 调用损失函数传入真实值和预测值，得到损失结果
            optimizer.zero_grad()   # 当前循环参数梯度置为0
            loss.backward() # 计算梯度
            optimizer.step()    更新参数的值
    matplotlib 支持tensor类型直接画图
"""
import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

# 定义数据
x = torch.rand([50, 1])
y = x * 3 + 0.8


# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        # input feature: 500 columns represents 1 feature
        # output feature: 500 columns represents 1 feature
        self.linear = nn.Linear(1, 1)

    def forward(self, x_):
        out_ = self.linear(x_)
        return out_


# 实例化模型
if __name__ == '__main__':
    model = LinearRegression()
    criterion = nn.MSELoss()
    # lr: learning rate
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    # 训练模型
    for i in range(30000):
        out = model(x)  # 获取预测值
        loss = criterion(y, out)  # 计算损失
        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 计算梯度
        optimizer.step()
        if (i + 1) % 200 == 0:
            print("Epoch[{}/{}], loss: {:.6f}".format(i, 30000, loss.data))

    # 模型评估
    model.eval()  # 设置模型为评估模式
    predict = model(x)
    predict = predict.data.numpy()
    plt.scatter(x, y, c='r')
    plt.plot(x, predict)
    plt.show()
