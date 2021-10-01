# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
import make_data

from test_function import *

"""
定义一个模型，来运用运用上面的方法
来一个2维向量2分类
"""
class demo_module(nn.Module):
    def __init__(self):
        super(demo_module,self).__init__()
        self.linear1 = nn.Linear(2,3)
#         self.h1 = nn.ReLU()
    def forward(self,x):
        y1 = self.linear1(x)
        return y1

def accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:#预测值y的是个矩阵且 类别大于1
        y_hat = y_hat.argmax(axis=1) #索引每组数值最大的下标
    cmp = y_hat.type(y.dtype) == y#因为==是很苛刻的，苛刻于两侧的类型必须一致。所以将预测值的类型转换和真实值一样 不然肯定是flase
    return float(cmp.type(y.dtype).sum())

if __name__ == '__main__':
    my_module = demo_module()

    lossfn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_module.parameters(), lr=0.1)

    epochs = 400
    n = 300
    train_data ,test_data = make_data.make_random_data(n)

    # print(list(zip(test_data[0],test_data[1])))

    for epoch in range(epochs):
        y_hat = my_module(train_data[0])
        y = train_data[1]
        # print('   :',accuracy(y_hat,y))
        loss = lossfn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

        #test
        print(evaluate_accuracy(my_module,list(zip(test_data[0],test_data[1]))))

