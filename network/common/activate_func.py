import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def tanh(x):
    exp_2x = np.exp(2 * x)
    return (exp_2x - 1) / (exp_2x + 1)

def identity(x):
    return x

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

if __name__ == '__main__':
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    print(softmax(x))