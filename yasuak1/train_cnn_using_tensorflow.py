from pkg_resources import add_activation_listener
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from disp import disp_acc

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
print(x_train.shape)
print(y_train.shape)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(60, 60, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(output_size, activation='softmax'))


model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )
result = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
metrics = ['loss', 'accuracy']

disp_acc(result=result, metrics=metrics)