
# coding: utf-8

# In[16]:


import numpy as np
import tensorflow as tf

arr1 = np.array([1, 3, 2, 5, 7, 0])

arr2 = np.array([[1,2,4], [3,4,9], [7,0,2], [8,3,2]])

print("r1=",arr1)
print("arr2=\n",arr2)

#找出最大值的下标
argmax_1 = tf.argmax(arr1)


#第二个参数为0，按第一维（的元素取值
#即从arr2[0][i],arr2[1][i],arr2[2][i],arr2[3][i]取出最大值，i从0 - 2
argmax_20 = tf.argmax(arr2,0)

#第二个参数为1，按第二维的元素取值
#即从arr2[i][0],arr2[i][1],arr2[2] 中取出最大值，i 从 0 - 3
argmax_21 = tf.argmax(arr2,1)
argmax_22 = tf.argmax(arr2,-1)

with tf.Session() as sess:
    print(argmax_1.eval())
    print(argmax_20.eval())
    print(argmax_21.eval())
    print(argmax_22.eval())

