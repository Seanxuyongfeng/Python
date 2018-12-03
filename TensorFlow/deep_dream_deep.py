
# coding: utf-8

# In[1]:

#导入Inception 模型

from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf

#创建会话
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

model_fn = 'tensorflow_inception_graph.pb' #导入Inception网络
#下载链接
#https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip

with tf.gfile.FastGFile(model_fn,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

#定义输入图像的占位符
t_input = tf.placeholder(np.float32,name='input')

#图像预处理---减均值
imagenet_mean = 117.0 #在训练Inception模型时做了减均值预处理，此处也需减同样的均值以保持一致

##图像预处理--增加维度
#图像数据格式一般是(height,width,channels)，为同时将多张图片输入网络而在前面增加一维
#变为(batch,height,width,channel)
t_preprossed = tf.expand_dims(t_input - imagenet_mean, 0)

#导入模型并将经预处理的属性送入网络中
tf.import_graph_def(graph_def,{'input':t_preprossed})


#找出卷积层
layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D']
#输出卷积层层数
print('Number of layers',len(layers))

#print(layers)

#还可以输出指定卷积层的参数
name1 = 'mixed4d_3x3_bottleneck_pre_relu'
print('shape of %s:%s' % (name1,str(graph.get_tensor_by_name('import/'+name1+':0').get_shape())))

name2 = 'mixed4d_5x5_bottleneck_pre_relu'
print('shape of %s:%s' % (name2,str(graph.get_tensor_by_name('import/'+name2+':0').get_shape())))

##卷积层格式一般是(batch,height,width,channel)，因为此时还不清楚输入图像的数量和大小，所以前三维是不确定的
#显示?
#shape of mixed4d_3x3_bottleneck_pre_relu:(?, ?, ?, 144)
#shape of mixed4d_5x5_bottleneck_pre_relu:(?, ?, ?, 32)

#由于导入的是已训练好的模型，所以指定的卷积层的通道数数量是固定的。



# In[15]:

#把一个numpy.ndarray保存成图像文件
def savearray(img_array,img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved:%s' % img_name)

#将图像放大ratio倍
def resize_ratio(img,ratio):
    min = img.min()
    max = img.max()
    img = (img-min)/(max-min)*255
    img = np.float32(scipy.misc.imresize(img,ratio))
    img = img/255 * (max - min) + min
    return img

#调整图像尺寸
def resize(img,hw):
    min = img.min()
    max = img.max()
    img = (img - min) /(max-min) * 255
    img = np.float32(scipy.misc.imresize(img,hw))
    img = img / 255 * (max - min) + min
    return img
#渲染函数
def render_naive(t_obj,img0,iter_n=20,step=1.0):
    #t_obj：是layer_output[:,:,:,channel]，即卷积层的某个通道
    #img0: 初始图像(噪声图像)
    #iter_n:迭代次数
    #step: 用于控制每次迭代步长，可视为学习率
    
    
    t_score = tf.reduce_mean(t_obj)
    #t_score是t_obj的平均值
    #由于我们的目标是调整输入图像使卷积层激活值尽可能大
    #即最大化t_score
    #为达到此目标，可使用梯度下降
    #计算t_score对t_input的梯度
    t_grad = tf.gradients(t_score,t_input)[0]
    
    img = img0.copy() #复制新图像可避免影响原始图像的值
    for i in range(iter_n):
        #在sess中计算梯度，以及当前的t_score
        g,score=sess.run([t_grad,t_score],{t_input:img})
        #对img应用梯度
        #首先对梯度进行归一化处理
        g/=g.std() + 1e-8
        #将正规化处理后的梯度应用在图像上，step用于控制每次迭代步长，此处为1.0
        img+=g*step
        print('iter:%d' %(i+1),'score(mean)=%f' % score)
    #保存图片
    savearray(img,'naive_deepdream.jpg')

#原始图像尺寸可能很大，从而导致内存耗尽的问题
#每次只对tile_size*tile_size大小的图像计算梯度，避免内存问题
def calc_grad_tiled(img,t_grad,tile_size=512):
    # tile_size:小图像的尺寸，就是将大图像分解为一个个小图像以避免内存耗尽问题
    sz = tile_size
    h,w = img.shape[:2]
    sx,sy=np.random.randint(sz,size=2)
    img_shift = np.roll(np.roll(img,sx,1),sy,0)#先在行上做整体移动，再在列上做整体移动
    grad = np.zeros_like(img)
    for y in range(0,max(h-sz//2,sz),sz):
        for x in range(0,max(w-sz//2,sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad,{t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad,-sx,1),-sy,0)
            
def render_deepdream(t_obj,img0,iter_n=10,step=1.5,octave_n=4,octave_scale=1.4):
    # octave_n：表示金字塔的层数
    # octave_scale：层与层之间的倍数，当乘以该octave_scale的时候，图像就是放大的
    #当除以octave_scale的时候，图像就是缩小的
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score,t_input)[0]
    img = img0.copy()
    
    ########################与render_naive不同部分############################
    #将图像进行金字塔分解
    #从而分为高频，低频部分
    octaves=[]
    for i in range(octave_n - 1):
        #hw表示缩放后，图像的高和宽
        hw = img.shape[:2]#获得图像的尺寸
        #得到低频成分
        lo = resize(img,np.int32(np.float32(hw)/octave_scale))
        #减去低频成分，得到高频成分
        hi = img - resize(lo,hw) #相减必须是同样的尺寸，所以需要resize为原图的尺寸
        img = lo
        #将高频成分保存在金字塔中
        octaves.append(hi)
    
    #首先生成低频的图像，再依次放大并加上高频
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img,hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img,t_grad)
            img += g*(step/(np.abs(g).mean()+1e-7))
    ###########################################################
    img = img.clip(0,255)
    savearray(img,'mountain_deepdream.jpg')
    im = PIL.Image.open('mountain_deepdream.jpg').show()


# In[16]:

#生成以背景图像作为起点的DeepDream图像

name = 'mixed4c'
layer_output = graph.get_tensor_by_name('import/%s:0'%name)

img0 = PIL.Image.open('mountain.jpg')
img0 = np.float32(img0)
render_deepdream(tf.square(layer_output),img0)

