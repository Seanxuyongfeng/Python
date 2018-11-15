
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pylab as plt
import numpy as np
from time import time
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#如果利用jupyter运行恢复代码，要注意In [*]  中的*，这根保存的时候有关，最好运行的时候能够重新restart一下， 使*从0开始

# mnist 中每张图片共有28*28个像素点
x = tf.placeholder(tf.float32, [None, 784], name="X")

# y是分类结果，一个结果又10个元素，代码是每个数字的概率
y = tf.placeholder(tf.float32, [None,10], name="Y")

#构建隐藏层
H1_NN = 256 #第1隐藏层神经元为256个
H2_NN = 64  #第2隐藏层神经元为64个
H3_NN = 32 #第3隐藏层盛景园为32个

############################################################################################
#定义全连接层函数
def fcn_layer(inputs,              #输入数据
              input_dim,          #输入神经元数量
              output_dim,          #输出神经元数量
              activation=None):    #激活函数
    #以截断正态分布的随机数初始化W
    W = tf.Variable(tf.truncated_normal([input_dim,output_dim], stddev=0.1))
    b = tf.Variable(tf.zeros([output_dim]))
    
    XWb = tf.matmul(inputs, W) + b
    
    if activation is None: #默认不适用激活函数
        outputs = XWb
    else:
        outputs = activation(XWb) #若传入激活函数，则用其对输出结果进行变换
    
    return outputs


#第1隐藏层参数和偏置项，标准差stddev=0.1
h1 = fcn_layer(inputs=x,
               input_dim=784,
               output_dim=H1_NN,
               activation=tf.nn.relu)


#第二隐藏层参数和偏置项
h2 = fcn_layer(inputs=h1,
               input_dim=H1_NN,
               output_dim=H2_NN,
               activation=tf.nn.relu)


#第三隐藏层参数和偏置项
h3 = fcn_layer(inputs=h2,
               input_dim=H2_NN,
               output_dim=H3_NN,
               activation=tf.nn.relu)


# 输出层参数和偏置项
forward = fcn_layer(inputs=h3,
               input_dim=H3_NN,
               output_dim=10,
               activation=None)

############################################################################################

#将结果forward进行分类化,总概率为1
pred = tf.nn.softmax(forward)

###
train_epochs = 40 #训练轮数
batch_size = 50 #每次训练样本数
total_batch = int(mnist.train.num_examples/batch_size) #一轮训练多少次
display_step = 1 #显示粒度
learning_rate = 0.01


#Tensorflow 提供了softmax_cross_entropy_with_logits 函数
#用于避免因为log(0) 值为NaN 造成的数据不稳定
#这里的forward 是没有经过softmax处理的数值
loss_function = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=forward,labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

# 检查预测类别tf.argmax(pred,1) 与实际类别tf.argmax(y,1) 的匹配情况
#这里也是批量的准确率correction_prediction，里面的值就是pred中的下标值，就是0-9之间的
#因为argmax 会取出pred每个中的最大值，也就是分出来的类别是什么
#相等为true，不相等为false
correction_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))


#将结果转换为float32类型，因为之前的结果是true，和false是无法直接进行运算的
#求这些prediction的平均值，作为准确率，也就是批量的预测的均值
#correct_prediction 转换之后，内容不是0就是1，就可以进行计算了
accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))


##################还原start###############
#必须指定为模型文件的存放目录
ckpt_dir = "./ckpt_dri/"

saver = tf.train.Saver()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

ckpt = tf.train.get_checkpoint_state(ckpt_dir)

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path) #从已保存的模型中读取数据
    print("Restore model from " + ckpt.model_checkpoint_path)
################## 还原 end##############

##############还原完成之后，在测试集上评估模型的准确率

accu_test = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})

print("Test Accuracy:",accu_test)

sess.close()

