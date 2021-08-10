# -*- coding:utf-8 -*-
"""
author: 15025
time: 10.08.2021   17:38:36
software: PyCharm

Description:
     requires_grad=True 将tensor进行的所有的操作都记录下来，方便后面进行操作
"""
import torch

# example about parameter requires_grad
x = torch.ones(2, 2, requires_grad=True)
# print(x)
y = x + 2
# print(y)
z = y * y * 3
# print(z)
out = z.mean()
# print(out)

# 输出结果
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)
# tensor([[27., 27.],
#         [27., 27.]], grad_fn=<MulBackward0>)
# tensor(27., grad_fn=<MeanBackward0>)
# 记录往回寻找时的操作：
#     AddBackward0： 加法
#     MulBackward0： 乘法
#     MeanBackward0： 均值

# 梯度: 反向传播算法计算梯度值,.grad属性输出个部分对应值
# 当输出作为一个标量时，可以直接使用backward()方法进行计算，但是如果不是标量，可能还需要传入其他参数
# 通常输出为一个损失函数，可能出现多次循环，每次都会对参数进行更新，所以当我们使用反向传播传播计算时，会将梯度累加到前一次的x.grad中，就会出错，因此我们需要在每次反向传播之前
# 需要先把梯度重置为0以后再进行计算
# out.backward()
# print(x.grad)

# ====================================================================================================================================================
# a = torch.randn(2, 2)
# a = (a * 3) / (a - 1)
# # 函数加()调用，属性值不加括号调用
# # print(a.requires_grad)
# a.requires_grad_(True)  # 就地修改
# # print(a.requires_grad)
# b = (a * a).sum()
# print(b.requires_grad)
# print(b)
# print(b.grad_fn)
# # 加with语句后，在其内书写的语句不会被追踪，该语句在评估模型中会适用，因为可能存在具有requires_grad = True的可训练参数
# with torch.no_grad():
#     c = (a * a).sum()
#
# print(c.requires_grad)


# ====================================================================================================================================================
# tensor.data()仅仅获取tensor中的数据，不带有属性 tensor, tensor.data()在当requires_grad=True时二者存在区别
# print(out)  # tensor(27., grad_fn=<MeanBackward0>)
# print(out.data)  # tensor(27.)

# 当tensor数据需要调用numpy操作时，最好使用tensor.detach().numpy()来进行转换操作
# .detach()相当于深拷贝，防止浅拷贝出错
