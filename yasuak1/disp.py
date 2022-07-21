from cProfile import label
from turtle import color
from matplotlib.pyplot import hist
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

def disp_acc(result, metrics):
    plt.figure()
    for i in range(len(metrics)):
        metric = metrics[i]
        
        plt_train = result.history[metric]
        plt_test = result.history['val_' + metric]

        plt.subplot(1, 2, i+1)
        plt.title(metric)
        plt.plot(plt_train, label='training')
        plt.plot(plt_test, label='test')
        plt.legend()
    plt.show()

def plot_image(i, predictions_array, true_label, class_names, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label: color = 'blue'
    else: color='red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100*np.max(predictions_array),
        class_names[true_label]),
        color=color
        )

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')