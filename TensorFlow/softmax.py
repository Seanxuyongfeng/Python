
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf

x = np.array([-3.1,1.8,9.7,-2.5])

pred = tf.nn.softmax(x)

# softmax ,其实就是让他们的结果处于0-1之间，且其和为1
# 算法为例如pred第一个元素的结果 e(-3.1)/(e(-3.1)+e(1.8)+e(9.7)+e(-2.5))
# 第二个元素结果 e(1.8)/(e(-3.1)+e(1.8)+e(9.7)+e(-2.5))
# 第三个元素结果 e(9.7)/(e(-3.1)+e(1.8)+e(9.7)+e(-2.5))
# 第四个元素结果 e(-2.5)/(e(-3.1)+e(1.8)+e(9.7)+e(-2.5))

#[  2.75972792e-06   3.70603254e-04   9.99621608e-01   5.02855213e-06]
sess = tf.Session()

v = sess.run(pred)

print(v)

sess.close()

