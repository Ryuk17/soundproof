"""
@ Filename:       cross_validation.py
@ Author:         Danc1elion
@ Create Date:    2019-09-20
@ Update Date:    2019-09-20
@ Description:    Implement cross_validation
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
from sklearn import metrics
import os

tf.set_random_seed(2019)

# add a layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    if activation_function is None:
        return tf.matmul(inputs, Weights), biases
    else:
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        outputs = activation_function(Wx_plus_b)
        return outputs

# F value
def calculateF(y_true, y_pred):
    y_pred = np.expand_dims(y_pred, axis=1)
    t = -103
    tp = len(y_true[(y_true < t) & (y_pred < t)])
    fp = len(y_true[(y_true >= t) & (y_pred < t)])
    fn = len(y_true[(y_true < t) & (y_pred >= t)])

    if tp + fp == 0 or tp + fn == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F = 2 * (precision * recall) / (precision * recall)
    return F

def calcaulateRMSE(y_true, y_pred):
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    return rmse

def draw(y_true, y_pred):
    x_data = range(len(y_true))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_true)
    lines = ax.plot(x_data, y_pred, 'r-', lw=5)
    fig.savefig("./test.png")
    plt.ion()
    plt.show()
    plt.pause(0.1)
    time.sleep(4)


# read data set
all_data = pd.read_csv('./data.csv')
all_label = all_data.pop('label')
all_data = np.array(all_data)
all_label = np.array(all_label)
all_label = np.expand_dims(all_label, axis=1)


feature_dim = all_data.shape[1]
input1 = all_data[:, :10]
input2 = all_data[:, 10:]

x1 = tf.placeholder(tf.float32, [None, input1.shape[1]], name = "haha_input_x1")
x2 = tf.placeholder(tf.float32, [None, input2.shape[1]], name = "haha_input_x2")
y_ = tf.placeholder(tf.float32, [None, 1])

# hyperparameter
n1 = 256
n2 = 128
n3 = 32

# create subnet1
layer11 = add_layer(x1, input1.shape[1], n1, activation_function=tf.nn.relu)
layer12 = add_layer(layer11, n1, n2, activation_function=tf.nn.relu)
layer13 = add_layer(layer12, n2, n3, activation_function=tf.nn.relu)

# create subnet2
layer21 = add_layer(x2, input2.shape[1], n1, activation_function=tf.nn.relu)
layer22 = add_layer(layer21, n1, n2, activation_function=tf.nn.relu)
layer23 = add_layer(layer22, n2, n3, activation_function=tf.nn.relu)

# concat subnet1 and subnet2
full_layer1 = tf.concat([layer13, layer23], axis=1)
full_layer2 = add_layer(full_layer1, 100, 50, activation_function=tf.nn.relu)
full_layer3 = add_layer(full_layer2, 50, 10, activation_function=tf.nn.relu)
w, b = add_layer(full_layer3, 10, 1)
y = tf.add(w, b, name = "haha_output_y")

# loss RMSE
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y)))
loss = tf.sqrt(loss)
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train
epoch = 1201
for i in tqdm(range(epoch)):
    sess.run(train_step, feed_dict={x1: input1, x2: input2, y_: all_label})
    if i % 200 == 0:
        y_pred = np.array(sess.run(y, feed_dict={x1: input1, x2:input2})).reshape(len(all_label))
        # draw(y_test, y_pred)
        print("Iteration %d, RMSE %f" % (i, calcaulateRMSE(all_label, y_pred)))

# save model
tf.saved_model.simple_save(sess,
            "./model1",
            inputs={"myInput1": x1, "myInput2": x2}, 
            outputs={"myOutput": y})
