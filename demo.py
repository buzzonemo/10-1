import numpy as np

n=100
X = np.random.randint(100,size = n)
Y = np.random.randint(200,size = n)

def F(x):
    return 2*x-3

def is_UporDown(*args):
    y_std = F(args[0])
    if args[1] > y_std:
        return 2
    elif args[1] == y_std:
        return 1
    elif args[1] < y_std:
        return 0


print(is_UporDown(*list(zip(X,Y))[0]))

print(list(zip(X,Y))[0])