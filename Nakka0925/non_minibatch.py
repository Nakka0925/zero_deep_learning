# coding: utf-8
import sys, os
import time
sys.path.append(os.pardir)

import numpy as np
from sklearn.model_selection import train_test_split
from datasets import data_gain
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
from common.optimizer import SGD, Adam

# データの読み込み
#(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
file_list, label_list = data_gain()

file_list = np.array(file_list)
label_list = np.array(label_list)
file_list = file_list.reshape(7550, 36864)


#0.0 ~ 1.0 に正規化
file_list = [file.astype(float)/255.0 for file in file_list]

x_train, x_test, t_train, t_test = train_test_split(file_list, label_list, test_size=0.2)

x_train = np.array(x_train)
t_train = np.array(t_train)
x_test = np.array(x_test)
t_test = np.array(t_test)

#print(t_train[:5])


network = TwoLayerNet(input_size=36864, hidden_size=100, output_size=3)

epoch = 30

learning_rate = 0.01

test_loss_list = []
train_loss_list = []
train_acc_list = []
test_acc_list = []

#iter_per_epoch = int(max(train_size / batch_size, 1))

start = time.time()
for i in range(epoch):
    #batch_mask = np.random.choice(train_size, batch_size)
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_train, t_train)
    
    
    #更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    

    #test = Adam()

    #network.params = test.update(network.params, grad)

    #if i % train_size == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_loss = network.loss(x_train, t_train)
    test_loss = network.loss(x_test, t_test)
    
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    print("train_accuracy:", train_acc, "test_accuracy:", test_acc)
    end = time.time()
    print("epoch" + str(int(i+1)), str(end - start) + 's')
    start = time.time()
    

# グラフの描画

x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker="o", label='train acc')
plt.plot(x, test_acc_list, marker="o", label='test acc',)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.savefig("accuracy_nonbatch.png")

plt.clf()

x = np.arange(len(train_acc_list))
plt.plot(x, train_loss_list, marker="o",label='train loss')
plt.plot(x, test_loss_list, marker="o", label='test loss',)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc='upper right')
plt.savefig("loss_nonbatch.png")