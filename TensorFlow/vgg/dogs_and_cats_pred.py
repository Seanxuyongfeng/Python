
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from scipy.misc import imread,imresize
import VGG16_model as model

means = [123.68,116.779,103.939]#VGG训练时图像预处理所减均值(RGB三通道)
x = tf.placeholder(tf.float32,[None,224,224,3])

sess = tf.Session()
vgg = model.vgg16(x)
fc8_finetuining = vgg.probs

saver = tf.train.Saver()
print("Model restoring...")
saver.restore(sess,'./model/')

filepath = './data/test1/test1/29.jpg'
img = imread(filepath,mode='RGB')
img = imresize(img,(224,224))
img = img.astype(np.float32)

for c in range(3):
    img[:,:,c] -= means[c]
prob = sess.run(fc8_finetuining,feed_dict={x:[img]})
max_index = np.argmax(prob)

if max_index == 0:
    print("This is a cat with possibility %.6f" %prob[:,0])
else:
    print("This is a dog with possibility %.6f" %prob[:,1])


