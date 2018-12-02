
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



# In[16]:

#把一个numpy.ndarray保存成图像文件
def savearray(img_array,img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved:%s' % img_name)

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



name = 'mixed4d_3x3_bottleneck_pre_relu' #(?,?,?,144)
# mixed4d_3x3_bottleneck_pre_relu 共有144个通道
#此处可选任意通道（0-143）之间任意整数，进行最大化
channel = 139

layer_output = graph.get_tensor_by_name('import/%s:0'%name)
#layout_output[:,:,:,channel] 即可表示该卷积层的第140个通道

#定义图像噪声
img_noise = np.random.uniform(size=(224,224,3)) + 100.0

#调整render_naive函数渲染
render_naive(layer_output[:,:,:,channel],img_noise,iter_n=20)

im = PIL.Image.open('naive_deepdream.jpg')
im.show()
im.save('naive_single_chn.jpg')


# In[10]:

#较低层单通道卷积特征生成DeepDream图像
#定义卷积层，通道数，并取出对应的tensor
name3 = 'mixed3a_3x3_bottleneck_pre_relu'
layer_output = graph.get_tensor_by_name('import/%s:0'%name3)
print('shape of %s:%s' %(name3,str(graph.get_tensor_by_name('import/'+name3+':0').get_shape())))
#shape of mixed3a_3x3_bottleneck_pre_relu:(?, ?, ?, 96)

#定义噪声图像
img_noise = np.random.uniform(size=(224,224,3)) + 100.0

#调用render_naive函数渲染
channel = 86 #(?,?,?,96)
render_naive(layer_output[:,:,:,channel],img_noise,iter_n=20)

im = PIL.Image.open('naive_deepdream.jpg')
im.show()
im.save('shallow_single_chn.jpg')


# In[15]:

#高层单通道卷积特征生成DeepDream图像
name4='mixed5b_5x5_pre_relu'
layer_output = graph.get_tensor_by_name('import/%s:0'%name4)
print('shape of %s:%s' %(name4,str(graph.get_tensor_by_name('import/'+name4+':0').get_shape())))
#shape of mixed5b_5x5_pre_relu:(?, ?, ?, 128)

#定义噪声图像
img_noise = np.random.uniform(size=(224,224,3)) + 100.0

#调用render_naive函数渲染
channel = 118 #(?,?,?,128)
render_naive(layer_output[:,:,:,channel],img_noise,iter_n=20)

im = PIL.Image.open('naive_deepdream.jpg')
im.show()
im.save('depp_single_chn.jpg')


# In[ ]:

##从上面的图片上阿奎那，从浅层 -- 高层，图像抽取的特征越来越抽象


# In[17]:

#生成原始DeepDream 图像---所有通道

name = 'mixed4d_3x3_bottleneck_pre_relu'
layer_output = graph.get_tensor_by_name('import/%s:0'%name)

#定义噪声图像
img_noise = np.random.uniform(size=(224,224,3)) + 100.0

#调用render_naive函数渲染
render_naive(layer_output,img_noise,iter_n=20)
#单通道时:layer_output[:,:,:,channel]

im = PIL.Image.open('naive_deepdream.jpg')
im.show()
im.save('all_chn.jpg')


# In[20]:

#以背景图像为起点生成deepdream图像
#之前的例子是以噪声图像为起点生成的deepdream图像
#下面就要以背景图片为起点生成deepdream图像

name = 'mixed4c'
layer_output = graph.get_tensor_by_name('import/%s:0'%name)

#之前的例子是以噪声图像为起点
#img_noise = np.random.uniform(size=(224,224,3)) + 100.0

#现在使用背景图片为起点
img_test = PIL.Image.open('mountain.jpg')

#调用render_naive函数渲染
render_naive(layer_output,img_test,iter_n=100)
#单通道时:layer_output[:,:,:,channel]

im = PIL.Image.open('naive_deepdream.jpg')
im.show()
im.save('mountain_naive.jpg')

