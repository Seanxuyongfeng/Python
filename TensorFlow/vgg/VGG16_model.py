
# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy as np
import os


#修改VGG模型：全连接层的神经元个数：trainable参数变动
#（1） 预训练的VGG是在ImageNet数据集上进行训练的，对1000个类别进行判定
#      若希望利用已训练模型用于其他分类任务，需要修改最后的全连接层
#（2） 在进行Finetuning（微调）对模型重新训练时，对于部分不需要训练的层可以通过设置trainable=False来确保训练过程中不会修改权值

class vgg16:
    def __init__(self, imgs):
        #在类初始化时加入全局列表，将需要共享的参数加载进来
        self.parameters = []
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        #将fc8作为模型的输出,输出每个属于各个类别的概率值
        self.probs = tf.nn.softmax(self.fc8)
    
    def saver(self):
        return tf.train.Saver()
    
    def fc_layers(self):
        self.fc6 = self.fc("fc1",self.pool5,4096,trainable=False)
        self.fc7 = self.fc("fc2",self.fc6,4096,trainable=False)
        #n_class是输出类别个数,fc8正是我们需要训练的
        self.fc8 = self.fc("fc3",self.fc7,2,trainable=True)
    
    def convlayers(self):
        #conv1
        self.conv1_1 = self.conv("conv1re_1",self.imgs,64,trainable=False)
        self.conv1_2 = self.conv("conv1_2",self.conv1_1,64,trainable=False)
        self.pool1 = self.maxpool("poolre1",self.conv1_2)
        
        #conv2
        self.conv2_1 = self.conv("conv2_1",self.pool1,128,trainable=False)
        self.conv2_2 = self.conv("convwe2_2",self.conv2_1,128,trainable=False)
        self.pool2 = self.maxpool("pool2",self.conv2_2)
        
        #conv3
        self.conv3_1 = self.conv("conv3_1",self.pool2,256)
        self.conv3_2 = self.conv("convrwe3_2",self.conv3_1,256,trainable=False)
        self.conv3_3 = self.conv("convrew3_3",self.conv3_2,256,trainable=False)
        self.pool3 = self.maxpool("poolre3",self.conv3_3)
        
        #conv4
        self.conv4_1 = self.conv("conv4_1",self.pool3,512,trainable=False)
        self.conv4_2 = self.conv("convrwe4_2",self.conv4_1,512,trainable=False)
        self.conv4_3 = self.conv("conv4rwe_3",self.conv4_2,512,trainable=False)
        self.pool4 = self.maxpool("pool4",self.conv4_3)
        
        #conv5
        self.conv5_1 = self.conv("conv5_1",self.pool4,512,trainable=False)
        self.conv5_2 = self.conv("convrwe5_2",self.conv5_1,512,trainable=False)
        self.conv5_3 = self.conv("conv5_3",self.conv5_2,512,trainable=False)
        self.pool5 = self.maxpool("poorwel5",self.conv5_3)
    
    def conv(self,name,input_data,out_channel,trainable=False):
        #对于不需要训练的层，因为是载入的参数，不希望该层的参数做改变，将trainable==Flase
        #input_data:输入数据
        #out_channel：输出通道数
        #获得输入数据的通道数
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            #卷积核
            kernel = tf.get_variable("weights",[3,3,in_channel,out_channel],dtype=tf.float32,trainable=False)
            biases = tf.get_variable("biases",[out_channel],dtype=tf.float32,trainable=False)
            conv_res = tf.nn.conv2d(input_data,kernel,[1,1,1,1],padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            #激活函数
            out = tf.nn.relu(res,name=name)
        #将卷积层定义的参数加入列表
        self.parameters += [kernel,biases]
        return out
    def fc(self,name,input_data,out_channel,trainable=False):
        #获取输入数据各个维度的维数
        shape = input_data.get_shape().as_list()
        if len(shape) == 4:
            #全连接层输入神经元个数size
            size = shape[-1] * shape[-2] * shape[-3]
        else:size = shape[1]
        #对数据进行展开
        input_data_flat = tf.reshape(input_data,[-1,size])
        with tf.variable_scope(name):
            weights = tf.get_variable("weights",shape=[size,out_channel],dtype=tf.float32,trainable=trainable)
            biases = tf.get_variable("biases",shape=[out_channel],dtype=tf.float32,trainable=trainable)
            res = tf.matmul(input_data_flat,weights)
            out = tf.nn.relu(tf.nn.bias_add(res,biases))
        #将全连接层定义的参数加入列表
        self.parameters += [weights, biases]
        return out
    def maxpool(self,name,input_data):
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding="SAME",name=name)
        return out
    
    def load_weights(self,weight_file,sess):
        #下载地址 
        #权重文件 https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
        #分类文件 https://www.cs.toronto.edu/~frossard/vgg16/imagenet_classes.py
        #该函数将获取的权重载入VGG模型中
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i,k in enumerate(keys):
            #剔除不需要载入的层
            if i not in [30,31]:
                sess.run(self.parameters[i].assign(weights[k]))
        print("----------------weights loaded------------------")



