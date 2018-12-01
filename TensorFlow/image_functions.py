
# coding: utf-8

# 图像编码处理

# In[1]:

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#读取图像原始数据
image_raw_data = tf.gfile.FastGFile("dog.jpg",'rb').read()

with tf.Session() as sess:
    #对图像进行jpeg格式解码从而得到图像对应的三维矩阵
    image_data = tf.image.decode_jpeg(image_raw_data)
    #image_data = tf.image.decode_png(image_raw_data)

    print(image_data.eval())
    
    plt.imshow(image_data.eval())
    plt.show()


# ## 图像缩放 tf.image.resize_images(images,new_height,new_width,method)

# ### 双线性插值法ResizeMethod.BILINEAR(默认设置),对应method=0

# In[2]:

with tf.Session() as sess:
    #用双线性插值法将图像缩放到新的尺寸
    resized1 = tf.image.resize_images(image_data,[256,256],method=0)
    #Tensorflow 的函数处理图片后存储数据是float32格式的，需要转换为uint8才能正确打印图片
    resized1 = np.asarray(resized1.eval(),dtype='uint8')
    plt.imshow(resized1)
    plt.show()


# ### 最邻近插值法NEAREST_NEIGHBOR,对应method=1
# ### 双立方插值法BICUBIC,对应method=2
# ### 像素区域插值法AREA,对应method=3

# ## 裁剪或填充后缩放
# ### tf.image.resize_image_with_crop_or_pad(image,target_height,target_height)
# 
# ### 如果目标图像尺寸小于原始图像尺寸，则在中心位置裁剪，反之则用黑色像素填充。

# In[3]:

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    croped = tf.image.resize_image_with_crop_or_pad(img_data,400,400)
    plt.imshow(croped.eval())
    plt.show()


# In[4]:

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    padded = tf.image.resize_image_with_crop_or_pad(img_data,2000,2000)
    plt.imshow(padded.eval())
    plt.show()


# ## 随机裁剪
# ### tf.image.random_crop(images,size,seed=None,name=None)

# In[6]:

import tensorflow as tf

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    random_cropped1 = tf.random_crop(img_data,[500,500,3])
    plt.imshow(random_cropped1.eval())
    plt.show()


# In[12]:

##再次随机裁剪，验证随机性
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    random_cropped1 = tf.random_crop(img_data,[500,500,3])
    plt.imshow(random_cropped1.eval())
    plt.show()


# In[14]:

#水平翻转 tf.image.flip_left_right(img_data)
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    plt.imshow(img_data.eval())
    plt.axis('off')
    plt.show()
    flip_left_right = tf.image.flip_left_right(img_data)
    plt.imshow(flip_left_right.eval())
    plt.axis('off')
    plt.show()


# In[15]:

##上下翻转
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    plt.axis('off')
    plt.imshow(img_data.eval())
    plt.show()
    flip_up_down = tf.image.flip_up_down(img_data)
    plt.imshow(flip_up_down.eval())
    plt.axis('off')
    plt.show()


# In[16]:

##改变对比度 tf.image.random_contrast

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    plt.imshow(img_data.eval())
    plt.show()
    #图像的对比度降为原来的二分之一
    contrast = tf.image.adjust_contrast(img_data,0.5)
    #将图像的对比度提高为原来的5倍
    #contrast = tf.image.adjust_contrast(img_data,5)
    #在[lower,upper]范围内随机调整图像对比度
    #contrast = tf.image.random_contrast(img_data,lower=0.2,uppper=3)
    plt.imshow(contrast.eval())
    plt.show()


# In[19]:

##白化处理
#将图像的像素值转化成零均值和单位方差
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    plt.imshow(img_data.eval())
    plt.show()
    
    standardization = tf.image.per_image_standardization(img_data)
    plt.imshow(np.asarray(standardization.eval(),dtype='uint8'))
    plt.show()

