# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

# LOAD DATA
cnt = 100
input_size = 480 * 640
train_path = '/home/mizuno/data/neural_data/train.npy'
test_path = '/home/mizuno/data/neural_data/test.npy'
train_dict = np.load(train_path, allow_pickle=True).item()
test_dict = np.load(test_path, allow_pickle=True).item()
print(test_dict)
x_train = train_dict['img_data']
y_train = train_dict['label']
x_test = test_dict['img_data']
y_test = test_dict['label']
"""
x_train = x_train.reshape(cnt, input_size)
x_test = x_test.reshape(cnt, input_size)

#print(x_train.shape)
#print(y_train.shape)

# Dropuoutの有無、割り合いの設定 ========================
use_dropout = True  # Dropoutなしのときの場合はFalseに
dropout_ratio = 0.5
# ====================================================

output_size = 2892
network = MultiLayerNetExtend(input_size=input_size, hidden_size_list=[100],
                              output_size=output_size, use_dropout=use_dropout, dropout_ration=dropout_ratio)
trainer = Trainer(network, x_train, y_train, x_test, y_test,
                  epochs=50, mini_batch_size=10,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
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
"""