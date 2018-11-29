
# coding: utf-8

# In[1]:

import urllib.request
import os
import tarfile

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

filepath = 'data/cifar-10-python.tar.gz'
if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.isfile(filepath):
    print(os.getcwd())
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)
else:
    print('Data file already exists.')

#解压
if not os.path.exists("data/cifar-10-batches-py"):
    tfile = tarfile.open("data/cifar-10-python.tar.gz",'r:gz')
    result = tfile.extractall("data/")
    print('Extracted to ./data/cifar-10-batches-py/')
else:
    print("Directory already exists.")


###https://www.cs.toronto.edu/~kriz/cifar.html

############下载完数据之后，开始加载数据#########


# In[2]:

############下载完数据之后，开始加载数据#########
import os
import numpy as np
import pickle as p

def load_CIFAR_batch(filename):
    '''load single batch of cifar'''
    with open(filename,'rb') as f:
        ##一个样本由标签和图像数据组成
        data_dict = p.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        
        #把原始数据结构调整为BCWH，这里第一参数是10000，因为一个finename中的数据是10000
        images = images.reshape(10000,3,32,32)
        #tensorflow处理图像数据的结构BWHC
        #因此需要把C也就是通道数据移动到最后一个元素，tanspose中的参数是index
        images = images.transpose(0,2,3,1)
        
        labels = np.array(labels)
        
        return images, labels

def load_CIFAR_data(data_dir):
    '''load cifar data'''
    images_train=[]
    labels_train=[]
    
    #这里5表示样本有五个文件data_batch_1，data_batch_2，data_batch_3，data_batch_4，data_batch_5
    for i in range(5):
        f = os.path.join(data_dir,'data_batch_%d' % (i+1))
        print('loading...', f)
        image_batch, label_batch = load_CIFAR_batch(f)
        images_train.append(image_batch)
        labels_train.append(label_batch)
        
        #转换成一维数组
        Xtrain = np.concatenate(images_train)
        Ytrain = np.concatenate(labels_train)
        del image_batch,label_batch
        
    ###加载测试集 test_batch
    Xtest,Ytest = load_CIFAR_batch(os.path.join(data_dir,'test_batch'))
    print('Finished loadding data!!')
    return Xtrain,Ytrain,Xtest,Ytest


data_dir='data/cifar-10-batches-py/'    
Xtrain,Ytrain,Xtest,Ytest = load_CIFAR_data(data_dir)


# In[3]:

#显示数据信息
print('traning image data shape:',Xtrain.shape)
print('traning label data shape:',Ytrain.shape)
print('test image data shape:',Xtest.shape)
print('test babel data shape:',Xtest.shape)


# In[4]:

#查看单项image和babel
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.imshow(Xtrain[11])
print(Ytrain[11])


# In[5]:

import matplotlib.pyplot as plt
#查看批量的iamges和babels
#https://www.cs.toronto.edu/~kriz/cifar.html
label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
           5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    flg = plt.gcf()
    flg.set_size_inches(12,6)
    if(num > 10):
        num = 10
    for i in range(0,num):
        ax = plt.subplot(2,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        
        title=str(i)+','+label_dict[labels[idx]]
        
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[idx]]
        
        ax.set_title(title,fontsize=10)
        
        idx+=1
    plt.show()

#批量查看测试集的图片和标签
plot_images_labels_prediction(Xtest,Ytest,[],0,10)


# In[6]:

#图像数据预处理

#查看图像数据信息
#显示第一个图的第一个像素点
print(Xtrain[0][0][0])

#将图像进行数字标准化
Xtrain_normalize = Xtrain.astype('float32')/255.0
Xtest_normalize = Xtest.astype('float32')/255.0

#查看预处理后的数据信息
print(Xtrain_normalize[0][0][0])

#查看标签数据
print(Ytrain[:10])


# In[7]:

###独热编码 One-Hot Encoding
###能够处理非连续数值特征
###在一定程度上也扩充了特征，比如性别本身是一个特征，听过one-hot编码以后，就变成了男或女两个特征
##就是把标签，比如原来是5，变为[0,0,0,0,0,1,0,0,0,0]
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)

yy=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]

encoder.fit(yy)

Ytrain_reshape = Ytrain.reshape(-1,1)
Ytrain_onehot = encoder.transform(Ytrain_reshape)
print("Ytrain_onehot shape:",Ytrain_onehot.shape)
print("Ytrain[0:5]",Ytrain[:5])
print("Ytrain_onehot[0:5]",Ytrain_onehot[:5])
Ytest_reshape =Ytest.reshape(-1,1)
Ytest_onehot = encoder.transform(Ytest_reshape)


# In[8]:

#######开始建立模型

import tensorflow as tf
tf.reset_default_graph()

#定义权重
def weight(shape):
    #在构建模型时，需要使用tf.Variable来创建一个变量
    #在训练时，这个变量不断更新
    #使用函数tf.truncated_normal(截断正态分布)生成标准差位0.1的随机数来初始化权重
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name='W')

def bias(shape):
    return tf.Variable(tf.constant(0.1,shape=shape),name='b')

#定义卷积操作，W就是卷积核
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#定义池化操作
#步长为2，即原尺寸的长和宽各除以2
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


###定义网络结构

#输入层，32x32图像，通道为3（RGB）
with tf.name_scope('input_layer'):
    x = tf.placeholder(tf.float32,shape=[None,32,32,3],name='x')

#第一个卷积层
#输入通道3，输出通道32，卷积后图像尺寸不变，依然是32x32
with tf.name_scope('conv_1'):
    W1 = weight([3,3,3,32])#[k_width,k_height,input_chn,output_chn]
    b1 = bias([32])
    conv_1 = conv2d(x,W1) + b1
    conv_1 = tf.nn.relu(conv_1)#非线性激活函数

#第一个池化层
#将32x32图像缩小为16x16，池化不改变通道数量，因此依然是32个
with tf.name_scope('pool_1'):
    pool_1 = max_pool_2x2(conv_1)

#第2个卷积层
#输入通道32，输出通道64，卷积后图像尺寸不变，依然是16x16
with tf.name_scope('conv_2'):
    W2 = weight([3,3,32,64])#[k_width,k_height,input_chn,output_chn]
    b2 = bias([64])
    conv_2 = conv2d(pool_1,W2) + b2
    conv_2 = tf.nn.relu(conv_2)#非线性激活函数

#第2个池化层
#将16x16图像缩小为8x8，池化不改变通道数量，因此依然是64个
with tf.name_scope('pool_2'):
    pool_2 = max_pool_2x2(conv_2)

#全连接层
#将第2个池化层的64个8x8的图像转化为一维向量，长度是64x8x8=4096
with tf.name_scope('fc'):
    W3= weight([4096,128])#定义128个神经元，可以调整
    b3=bias([128])
    flat = tf.reshape(pool_2,[-1,4096])
    h = tf.nn.relu(tf.matmul(flat,W3)+b3)
    h_dropout=tf.nn.dropout(h,keep_prob=0.8)#用来避免过拟合

#输出层，输出层共有10个神经元，10分类问题，对应0-9
with tf.name_scope('output_layer'):
    W4=weight([128,10])
    b4=bias([10])
    pred = tf.nn.softmax(tf.matmul(h_dropout,W4)+b4)

####构建模型
with tf.name_scope('optimizer'):
    y = tf.placeholder('float',shape=[None,10],name='label')
    loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)

#定义准确率
with tf.name_scope('evaluation'):
    correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))



# In[9]:

####启动会话
import os
from time import time

train_epochs = 25
batch_size = 50
total_batch = int(len(Xtrain)/batch_size)
epoch_list=[]
accuracy_list=[]
loss_list=[]


epoch = tf.Variable(0,name='epoch',trainable=False)

startTime = time()
sess = tf.Session()
init= tf.global_variables_initializer()
sess.run(init)


# In[22]:

##断点续训
ckpt_dir = 'CIFAR10_log/'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

saver = tf.train.Saver(max_to_keep=1)

#如果有检查点文件，读取最新的检查点文件，恢复各种变量值
ckpt=tf.train.latest_checkpoint(ckpt_dir)
if ckpt != None:
    saver.restore(sess,ckpt)#加载所有的参数
    #从这里开始就可以直接使用模型进行预测，或者接着继续训练了
else:
    print('Training from scratch.')

#获取之前训练的轮次，这次从start次开始
start = sess.run(epoch)
print('Training starts from {} epoch.'.format(start+1))


# In[23]:

##迭代训练
def get_train_batch(number,batch_size):
    return Xtrain_normalize[number*batch_size:(number+1)*batch_size],                     Ytrain_onehot[number*batch_size:(number+1)*batch_size]

for ep in range(start, train_epochs):
    for i in range(total_batch):
        batch_x,batch_y = get_train_batch(i,batch_size)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if i%100 == 0:
            print("Step {}".format(i), "finished.")
    
    loss,acc = sess.run([loss_function,accuracy],feed_dict={x:batch_x,y:batch_y})
    
    epoch_list.append(ep+1)
    loss_list.append(loss)
    accuracy_list.append(acc)
    
    print("Train epoch:",'%02d' % (sess.run(epoch)+1),"Loss=","{:.6f}".format(loss),"Accuracy=",acc)
    
    saver.save(sess,ckpt_dir+"CIFAR10_cnn_model.ckpt",global_step=ep+1)
    sess.run(epoch.assign(ep+1))

duration = time() - startTime
print("Train finished takes:",duration)


# In[24]:

##可视化损失值
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

flg = plt.gcf()
flg.set_size_inches(4,2)
plt.plot(epoch_list,loss_list,label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'],loc='upper right')


# In[25]:

###可视化准确率

plt.plot(epoch_list,accuracy_list,label='accuracy')
flg = plt.gcf()
flg.set_size_inches(4,2)
plt.ylim(0.1,1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()


# In[28]:

#计算测试集上的准确率
test_total_batch = int(len(Xtest_normalize)/batch_size)
test_acc_sum = 0.0
for i in range(test_total_batch):
    test_image_batch = Xtest_normalize[i*batch_size:(i+1)*batch_size]
    test_label_batch = Ytest_onehot[i*batch_size:(i+1)*batch_size]
    test_batch_acc = sess.run(accuracy,feed_dict={x:test_image_batch,y:test_label_batch})
    test_acc_sum += test_batch_acc

test_acc = float(test_acc_sum/test_total_batch)
print("Test accuracy:{:.6f}".format(test_acc))


# In[31]:

#利用模型进行预测
test_pred = sess.run(pred,feed_dict={x:Xtest_normalize[:10]})
prediction_result=sess.run(tf.argmax(test_pred,1))

#可视化预测结果
plot_images_labels_prediction(Xtest,Ytest,prediction_result,0,10)

