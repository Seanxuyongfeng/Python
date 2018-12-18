
# coding: utf-8

# In[ ]:

import os
import tensorflow as tf
from time import time
import VGG16_model as model
import utils

startTime = time()
batch_size = 16
capacity=256 #内存中存储的最大数据容量
mean = [123.68,116.779,103.939]#VGG训练时图像预处理所减均值(RGB三通道)

#获取图像列表和标签列表
xs,ys = utils.get_file_forwindows("./data/train/")
print("xs len:",len(xs))
print("ys len:",len(ys))
image_batch,label_batch = utils.get_batch(xs,ys,224,224,batch_size,capacity)

x = tf.placeholder(tf.float32,[None,224,224,3])
y = tf.placeholder(tf.int32,[None,2])#对 猫 和 狗两个类别进行判定

vgg = model.vgg16(x)
fc8_finetuining = vgg.probs #既softmax(fc8)
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc8_finetuining,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_function)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
vgg.load_weights('./vgg16_weights.npz',sess)
saver = tf.train.Saver()

##启动线程
coord = tf.train.Coordinator() #使用协调器Coordinator来管理线程
threads = tf.train.start_queue_runners(coord=coord, sess=sess)


epoch_start_time = time()

for i in range(1000):
    images,labels = sess.run([image_batch,label_batch])
    labels = utils.onehot(labels)
    
    sess.run(optimizer, feed_dict={x:images,y:labels})
    loss = sess.run(loss_function,feed_dict={x:images,y:labels})
    print("Now the loss is %f" %loss)
    
    epoch_end_time = time()
    print('Current epoch takes:',(epoch_end_time - epoch_start_time))
    epoch_start_time = epoch_end_time
    
    if (i+1) % 500 == 0:
        saver.save(sess,os.path.join("./model",'epoch_{:06d}.ckpt'.format(i)))
    print("------------------Epoch %d is finished------------------"%i)

#模型保存
saver.save(sess,"./model/")
print("Optimization Finished!")

duration = time() - startTime
print("Train Finished takes:","{:.2f}".format(duration))


#关闭线程
coord.request_stop()#通知其他线程关闭
#join操作等待其他线程结束，其他所有线程关闭之后，这一函数才返回
coord.join(threads)

