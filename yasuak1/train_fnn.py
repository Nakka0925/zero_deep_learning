# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

# LOAD DATA
input_size = 480 * 640
train_path = '/home/mizuno/data/neural_data/train.pkl'
test_path = '/home/mizuno/data/neural_data/test.pkl'
train_dict = pd.read_pickle(train_path, compression="zip")
test_dict = pd.read_pickle(test_path, compression="zip")
x_train, y_train = train_dict['img_data'], train_dict['label']
x_test, y_test = test_dict['img_data'], test_dict['label']
train_size = x_train.shape[0]
test_size = y_test.shape[0]

x_train = x_train.reshape(train_size, input_size)
x_test = x_test.reshape(test_size, input_size)

y_train = np.array(y_train, dtype=int)
y_test = np.array(y_test, dtype=int)

print(set(y_train))
print(set(y_test))

# Dropuoutの有無、割り合いの設定 ========================
use_dropout = True  # Dropoutなしのときの場合はFalseに
dropout_ratio = 0.0
# ====================================================

output_size = 2892
network = MultiLayerNetExtend(input_size=input_size, hidden_size_list=[100],
                              output_size=output_size, use_dropout=use_dropout, dropout_ration=dropout_ratio)
trainer = Trainer(network, x_train, y_train, x_test, y_test,
                  epochs=5, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.001}, verbose=True)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# グラフの描画==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()