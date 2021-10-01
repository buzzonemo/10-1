import torch
import torch.nn as nn

def accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:#预测值y的是个矩阵且 类别大于1
        y_hat = y_hat.argmax(axis=1) #索引每组数值最大的下标
    cmp = y_hat.type(y.dtype) == y#因为==是很苛刻的，苛刻于两侧的类型必须一致。所以将预测值的类型转换和真实值一样 不然肯定是flase
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """在`n`个变量上累加。"""
    def __init__(self, n):
        '''
        初始化对象的时候，初始有多少个内容需要被累加
        '''
        self.data = [0.0] * n

    def add(self, *args):
        '''
        args元组的数量和self.data开辟的空间数量是一样的，self.data中的每一个空间，都用于存储args对应的数值
        这里存储的内容是：self.data[0] 存储 args[0] 的第一个的累加内容
        '''
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        '''
        魔法方法，用于给实例化的对象索引
        '''
        return self.data[idx]



def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel()) #numel是tensor这种数据类型的专有方法，统计张量的数量
    return metric[0] / metric[1]