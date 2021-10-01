import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
'''
传入参数：二维数据
输出结果：判断在目标二维线性函数的上方还是下方
'''
def F(x):
    return 2*x-3
def is_UporDown(*args):
    y_std = F(args[0])
    if args[1] > y_std:
        return 1
    elif args[1] == y_std:
        return 0
    elif args[1] < y_std:
        return -1

'''
输入：样本个数n
输出：训练集、测试集
'''
def make_random_data(n):
    slice = int(n*(2/3))
    X = np.random.randint(100, size=n)
    Y = np.random.randint(200, size=n)

    Data = list(zip(X,Y))


    list_class = np.array([])
    for data in Data:
        list_class = np.append(list_class, is_UporDown(*data) + 1)

    Data_Train = torch.tensor(Data[:slice], dtype=torch.float32)
    Data_Test = torch.tensor(Data[slice:], dtype=torch.float32)

    Data_Y = torch.LongTensor(list_class)
    # print(Data_Y[:slice])
    return [Data_Train,Data_Y[:slice]],[Data_Test,Data_Y[slice:]]

def dd(*args):
    print(args[0])

if __name__ == '__main__':
    d,t = make_random_data(300)
    print(d)

