# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datasets import data_gain
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

file_list, label_list = data_gain()

file_list = np.array(file_list)
label_list = np.array(label_list)
file_list = file_list.reshape(7550, 1, 192, 192)

#0.0 ~ 1.0 に正規化
file_list = [file.astype(float)/255.0 for file in file_list]

train_x, valid_x, train_y, valid_y = train_test_split(file_list, label_list, test_size=0.2)

train_y = np.array(train_y)
valid_y = np.array(valid_y)

train_x = np.array(train_x)
valid_x = np.array(valid_x)


"""
x_val = valid_x[:755]
y_val = valid_y[:755]

test_x = valid_x[755:]
test_y = valid_y[755:]
"""
max_epochs = 10

# Dropuoutの有無、割り合いの設定 ========================
use_dropout = True  # Dropoutなしのときの場合はFalseに
dropout_ratio = 0.5
# ====================================================

network = SimpleConvNet(input_dim=(1,192,192), 
                        conv_param = {'filter_num': 8, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=3, weight_init_std='Relu', use_dropout = use_dropout,
                        dropout_ration = dropout_ratio)
                        
trainer = Trainer(network, train_x, train_y, valid_x, valid_y, 
                  epochs=max_epochs, mini_batch_size = 32,
                  optimizer='Adam', optimizer_param={'lr': 0.001}
                  )
trainer.train()

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
#markers = {'train': 'o', 'test': 's'}
x = np.arange(1, max_epochs+1)
plt.plot(x, trainer.train_acc_list, marker='o', label='train')
plt.plot(x, trainer.test_acc_list, marker='o', label='test')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0.5, 1.0)
plt.legend(loc='lower right')
plt.savefig("accuracy_Cov_batch32.png")

plt.clf()

plt.plot(x, trainer.train_loss_list, marker='o', label='train')
plt.plot(x, trainer.test_loss_list, marker='o', label='test')
plt.xlabel("epochs")
plt.ylabel("loss")
#plt.ylim(0, 2.0)
plt.legend(loc='upper right')
plt.savefig("loss_Cov_batch32.png")