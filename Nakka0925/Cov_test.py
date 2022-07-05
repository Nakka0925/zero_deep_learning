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

print(file_list.shape)

#0.0 ~ 1.0 に正規化
file_list = [file.astype(float)/255.0 for file in file_list]

x_train, x_test, t_train, t_test = train_test_split(file_list, label_list, test_size=0.2)

x_train = np.array(x_train)
t_train = np.array(t_train)
x_test = np.array(x_test)
t_test = np.array(t_test)


max_epochs = 5

network = SimpleConvNet(input_dim=(1,192,192), 
                        conv_param = {'filter_num': 32, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=3, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
#markers = {'train': 'o', 'test': 's'}
x = np.arange(len(trainer.train_acc_list))
plt.plot(x, trainer.train_acc_list, marker='o', label='train')
plt.plot(x, trainer.test_acc_list, marker='o', label='test')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.savefig("acuracy_Cov.png")