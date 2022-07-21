import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

size = 60
train_path = '/home/yasu/data/neural_data/train.pkl'
test_path = '/home/yasu/data/neural_data/test.pkl'
train_dict = pd.read_pickle(train_path, compression="zip")
test_dict = pd.read_pickle(test_path, compression="zip")
x_train, y_train = train_dict['img_data'], train_dict['label']
x_test, y_test = test_dict['img_data'], test_dict['label']
train_size = x_train.shape[0]
test_size = x_test.shape[0]
label = {}
cnt = 0
new_key = 0
for key in set(y_train):
    label[key] = cnt
    cnt += 1
for key in set(y_test):
    if not key in label:
        label[key] = cnt
        cnt += 1

new_y_train = np.empty(0)
new_y_test = np.empty(0)

for key in y_train:
    new_y_train = np.append(new_y_train, label[key])
for key in y_test:
    new_y_test = np.append(new_y_test, label[key])
y_train = new_y_train
y_test = new_y_test

output_size = len(label)

x_train = 1 - x_train.reshape(train_size, size, size, 1)
x_test = 1 - x_test.reshape(test_size, size, size, 1)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(size, size)), 
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(output_size, activation='softmax')
])
model.compile(
  optimizer='adam', 
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=100)

metrics = ['loss', 'accuracy']
plt.figure()
for i in range(len(metrics)):
 
    metric = metrics[i]
 
    plt.subplot(1, 2, i+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
    plt.title(metric)  # グラフのタイトルを表示
    
    plt_train = history.history[metric]  # historyから訓練データの評価を取り出す
    plt_test = history.history['val_' + metric]  # historyからテストデータの評価を取り出す
    
    plt.plot(plt_train, label='training')  # 訓練データの評価をグラフにプロット
    plt.plot(plt_test, label='test')  # テストデータの評価をグラフにプロット
    plt.legend()  # ラベルの表示
    
plt.show()  # グラフの表示