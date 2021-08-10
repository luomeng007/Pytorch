# -*- coding:utf-8 -*-
"""
author: 15025
time: 10.08.2021   16:53:09
software: PyCharm

Description:

"""

# 使用pytorch手动来完成线性回归
import torch
import matplotlib.pyplot as plt

learning_rate = 0.01  # 学习率

# 1. 准备数据
# y = 3x + 0.8
x = torch.rand([500, 1])
y_true = x * 3 + 0.8

# 2. 通过模型计算y_predict
w = torch.rand([1, 1], requires_grad=True)
b = torch.tensor(0, requires_grad=True, dtype=torch.float32)  # 偏差

# 4. 通过循环，反向传播，更新参数
for i in range(2000):
    y_predict = torch.matmul(x, w) + b
    # 3. 计算loss, 方差
    loss = (y_true - y_predict).pow(2).mean()
    if w.grad is not None:
        w.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()
    loss.backward()
    w.data = w.data - learning_rate * w.grad
    b.data = b.data - learning_rate * b.grad
    if i % 50 == 0:
        print("w, b, loss", w.item(), b.item(), loss.item())

plt.figure(figsize=(16, 8))
plt.scatter(x.numpy().reshape(-1), y_true.numpy().reshape(-1))
y_predict = torch.matmul(x, w) + b
plt.plot(x.numpy().reshape(-1), y_predict.detach().numpy().reshape(-1), c="r")
plt.show()