
# coding: utf-8

# In[15]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pylab as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print('训练集 train 数量: ', mnist.train.num_examples,
     '验证集 validation 数量: ', mnist.validation.num_examples,
     '测试集 test 数量: ', mnist.test.num_examples)

print('train images shape:', mnist.train.images.shape,
     'labels shape:', mnist.train.labels.shape)

# 784,一个图像
len(mnist.train.images[0])

print(mnist.train.images[0])

mnist.train.images[0].reshape(28,28)

##################图像显示####################

def plot_image(image):
    plt.imshow(image.reshape(28,28),cmap='binary')
    plt.show()

plot_image(mnist.train.images[0])


# In[ ]:




