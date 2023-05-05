# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X, Y = read_data()
    Z = -0.1
    WEIGHT = np.dot(np.linalg.inv((np.dot(X.T, X) + np.dot(Z, np.eye(6)))), np.dot(X.T, Y))
    return WEIGHT @ data
    
def lasso(data):
    X, y = read_data()
    WEIGHT = data
    Y = np.dot(WEIGHT, X.T)
    Z = 4000
    RATE = 1e-11
    for i in range(int(2e5)):
        Y = np.dot(WEIGHT, X.T)
        W = np.dot(Y - y, X) + Z * np.sign(WEIGHT)
        WEIGHT = WEIGHT * (1 - (RATE * Z / 6)) - W * RATE
    return WEIGHT @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
