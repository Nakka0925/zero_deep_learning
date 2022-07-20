# coding: utf-8
from imghdr import tests
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from datasets import data_gain
from simple_convnet import SimpleConvNet
from common.trainer import Trainer


file_list, label_list = data_gain()

file_list = np.array(file_list)
label_list = np.array(label_list)
file_list = file_list.reshape(7550, 1, 192, 192)

#0.0 ~ 1.0 に正規化
file_list = [file.astype(float)/255.0 for file in file_list]


train_x, test_x, train_y, test_y = train_test_split(file_list, label_list, test_size=0.2,
                                                    stratify = label_list)


train_x = np.array(train_x)
test_x = np.array(test_x)

train_y = np.array(train_y)
test_y = np.array(test_y)

#層化抽出法を用いたK-分割交差検証
splits = 5
kf = StratifiedKFold(n_splits=splits, shuffle=True)
all_loss = []
all_val_loss = []
all_acc = []
all_val_acc = []

max_epochs = 10

# Dropuoutの有無、割り合いの設定 ========================
use_dropout = True  # Dropoutなしのときの場合はFalseに
dropout_ratio = 0.5
# ====================================================

for train_index, val_index in kf.split(train_x, train_y):              
    
    train_data  = train_x[train_index]
    train_label = train_y[train_index]
    val_data    = train_x[val_index]
    val_label   = train_y[val_index]

    network = SimpleConvNet(input_dim=(1,192,192), 
                        conv_param = {'filter_num': 8, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=64, output_size=3, weight_init_std='Relu', use_dropout = use_dropout,
                        dropout_ration = dropout_ratio)
    
    trainer = Trainer(network, train_data, train_label, val_data, val_label, test_x, test_y,
                  epochs=max_epochs, batch_size = 128,
                  optimizer='Adam', optimizer_param={'lr': 0.001}
                  )
    trainer.train()

    acc = trainer.train_acc_list
    val_acc = trainer.test_acc_list
    loss = trainer.train_loss_list
    val_loss = trainer.test_loss_list

    all_acc.append(acc)
    all_val_acc.append(val_acc)
    all_loss.append(loss)
    all_val_loss.append(val_loss)


#accuracy, loss平均の平均
ave_all_acc = np.mean(all_acc, axis = 0)
ave_all_val_acc = np.mean(all_val_acc, axis = 0)
ave_all_loss = np.mean(all_loss, axis = 0)
ave_all_val_loss = np.mean(all_val_loss, axis = 0)
print(ave_all_acc)
print(ave_all_loss)

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
#markers = {'train': 'o', 'test': 's'}
x = np.arange(1, max_epochs+1)
plt.plot(x, ave_all_acc, marker='o', label='train')
plt.plot(x, ave_all_val_acc, marker='o', label='val')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0.5, 1.0)
plt.legend(loc='lower right')
plt.savefig("accuracy_Cov_Cross-validation.png")

plt.clf()

plt.plot(x, ave_all_loss, marker='o', label='train')
plt.plot(x, ave_all_val_loss, marker='o', label='val')
plt.xlabel("epochs")
plt.ylabel("loss")
#plt.ylim(0, 2.0)
plt.legend(loc='upper right')
plt.savefig("loss_Cov_Cross-validation.png")