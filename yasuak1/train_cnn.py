# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from simple_conv import SimpleConvNet
from common.trainer import Trainer

w_size = 120
h_size = 120
input_size = h_size * w_size
train_path = '/home/mizuno/data/neural_data/train.pkl'
test_path = '/home/mizuno/data/neural_data/test.pkl'
train_dict = pd.read_pickle(train_path, compression="zip")
test_dict = pd.read_pickle(test_path, compression="zip")
x_train, y_train = train_dict['img_data'], train_dict['label']
x_test, y_test = test_dict['img_data'], test_dict['label']
train_size = x_train.shape[0]
test_size = y_test.shape[0]

#output_size = max(len(set(y_train)), len(set(y_test)))
output_size = 2892

x_train = x_train.reshape(train_size, 1, h_size, w_size)
x_test = x_test.reshape(test_size, 1, h_size, w_size)

x_train = 1 - x_train
x_test = 1 - x_test

y_train = np.array(y_train, dtype=int)
y_test = np.array(y_test, dtype=int)


tran_key = {}
cnt = 0
for key in set(y_train):
    tran_key[key] = cnt
    cnt += 1
for key in set(y_test):
    if not key in tran_key:
        tran_key[key] = cnt
        cnt += 1
buf_y_train = np.empty(0)
buf_y_test = np.empty(0)
for key in y_train:
    buf_y_train = np.append(buf_y_train, tran_key[key])
for key in y_test:
    buf_y_test = np.append(buf_y_test, tran_key[key])
y_train = buf_y_train
y_test = buf_y_test

y_train = np.array(y_train, dtype=int)
y_test = np.array(y_test, dtype=int)
print(y_train)

print(x_train.shape)
print(set(y_train))
print(set(y_test))

max_epochs = 20

network = SimpleConvNet(input_dim=(1, h_size, w_size), 
                        conv_param = {'filter_num': 10, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=output_size, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, y_train, x_test, y_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.01},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
