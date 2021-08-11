# -*- coding:utf-8 -*-
"""
File: T01_basis.py
author: Diligent Big Panda
date: 09.08.2021   20:28
Description:

"""
import torch
import numpy as np

# create a tensor
# t1 = torch.Tensor([1, 2, 3])
# print(t1)

# ======================================================================================================================
# transfer numpy array to tensor
# array1 = np.arange(12).reshape(3, 4)
# t1 = torch.Tensor(array1)
# print(t1)
# ======================================================================================================================
# create empty
# t1 = torch.empty([3, 4])
# print(t1)

# ======================================================================================================================
# t1 = torch.rand([3, 4])
# print(t1)

# ======================================================================================================================
# create int type tensor
# t1 = torch.randint(low=0, high=3, size=[3, 4])
# print(t1)

# ======================================================================================================================
# 获取tensor中的数据，当tensor中只有一个元素可用时tensor.item()
# t1 = torch.tensor(1)
# print(t1.item())

# ======================================================================================================================
# transfer tensor to numpy array
# t1 = torch.tensor([1, 2])
# print(t1.numpy())

# ======================================================================================================================
# size method: get shape of tensor
# t1 = torch.tensor([[[1, 2, 3]]])
# print(t1.size())
# print(t1.size(0))
# print(t1.size(1))
# print(t1.size(2))

# ======================================================================================================================
# change shape
# t1 = torch.Tensor([[[1, 2],
#                     [2, 3],
#                     [3, 4]]])
# print(t1.view(-1))
# print(t1.view(2, 3))
# print(t1.view(2, -1))
# print(t1.view(3, 2))

# ======================================================================================================================
# t1 = torch.Tensor([[[1, 2],
#                     [2, 3],
#                     [3, 4]]])
# # dim method
# print(t1.dim())
#
# # max method
# print(t1.max())
#
# # min method
# print(t1.min())
#
# # standard deviation
# print(t1.std())

# ======================================================================================================================
# transpose permute
# 1-D case, still be origin tensor
# t1 = torch.tensor(1)
# print(f"The original tensor is: {t1}")
# print(f"The transpose tensor of t1 is {t1.t()}")
# print(f"The transpose tensor of t1 is {torch.transpose(input=t1, dim0=0, dim1=-1)}")

# 2-D case
# t1 = torch.tensor([[1, 2], [3, 4]])
# print(f"The original tensor is: {t1}")
# print(f"The transpose tensor of t1 is {t1.t()}")
# print(f"The transpose tensor of t1 is {torch.transpose(input=t1, dim0=0, dim1=1)}")

# t1.t()仅适用于0维和1维，0维返回的是它本身，1维返回的是转置矩阵
# 高阶张量存在转置
# torch.transpose(input=t1, dim0=2, dim1=1)可以将高阶张量的任意两个方向的进行转置,但是一次只能实现两方向之间的转置
# t1 = torch.tensor(np.ones([2, 2, 2]))
# # print(torch.transpose(input=t1, dim0=2, dim1=1))
# print(t1.T)  # 等同于tensor.permute(n-1, n-2 .... 0)

# tensor.permute()实现多行之间的转置

# ======================================================================================================================
# t1 = torch.tensor(np.arange(24).reshape((2, 3, 4)))
# get single value
# print(t1[1, 2, 1])
# get first block
# print(t1[0, :, :])

# ======================================================================================================================
# default type: torch.int32
# t1 = torch.tensor(np.arange(24).reshape((2, 3, 4)))
# print(t1.dtype)

# ======================================================================================================================
# 指定输入数据类型
# t1 = torch.tensor(1, dtype=torch.double)
# print(t1)

# t1 = torch.tensor(np.array(12, dtype=np.int32))
# print(t1)
# print(t1.dtype)

# t1 = torch.LongTensor(1, 2)
# print(t1)
#
# t1 = torch.DoubleTensor(1, 2)
# print(t1)
# ======================================================================================================================
# t1 = torch.tensor(data=()).new_ones(3, 5)
# # print(t1)
# t2 = torch.rand((3, 5))
# # add method
# # print(t1 + t2)
# # print(torch.add(t1, t2))
#
# # 带有下划线的方法，指代就地修改，计算完成后对操作的tensor变量进行修改
# t1.add_(t2)
# print(t1)

# ======================================================================================================================
# 使tensor能够在cuda gpu中运行的方法, this computer does not support GPU
# if torch.cuda.is_available():   # judge whether current computer support GPU or not.
#     print(1)

# ======================================================================================================================
