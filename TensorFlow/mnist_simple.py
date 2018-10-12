from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#################################################
#print全部输出
numpy.set_printoptions(threshold=numpy.inf)  
# 输入数据，每一张图片有784个像素点 ，用一个2阶张量表示,[[...],[...],[...],[784个元素...]]
x = tf.placeholder("float", [None, 784])

# 初始化所有的权重W，每一个像素点都对应10个权重w，一张图片(28x28) 有784个像素,[[10个元素...],[...],[...],[...]]
W = tf.Variable(tf.zeros([784, 10]))

# 对于一张图片有10个输出，也就是10个数字可能，对应10个篇置量
b = tf.Variable(tf.zeros([10]))

# 创建模型，计算每张图片对应的10个数字的可能性大小
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义一个占位符，用于输入正确值，每张图片有10个可能的值，这个正确的值用于后面的模型训练
y_ = tf.placeholder("float", [None, 10])

# 创建损失模式，交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 使用梯度下降算法以 0.01的学习率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


init = tf.global_variables_initializer()

sess = tf.Session()
# 初始化所有变量
sess.run(init)

# 每次循环抓取训练数据中的1--个批处理数据点，利用这些数据点替换之前定义的占位符来运行
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 至此训练完毕，打印一下训练出的w和b
print('W: %s ' %(sess.run(W)))
print('b : %s' %(sess.run(b)))

##############评估训练模型

### 这里的y是之前定义的 y = tf.nn.softmax(tf.matmul(x, W) + b)，其中W 和b 是已经训练出来了的值，故可以直接使用y
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

### 对于比较的结果 取平均值,也就是正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

###在测试数据集上的正确率,在这里数据集就换成了mnist.test
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))