
# coding: utf-8

# In[12]:

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


# In[19]:

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


# In[20]:

#显示数据信息
print('traning image data shape:',Xtrain.shape)
print('traning label data shape:',Ytrain.shape)
print('test image data shape:',Xtest.shape)
print('test babel data shape:',Xtest.shape)


# In[25]:

#查看单项image和babel
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.imshow(Xtrain[11])
print(Ytrain[11])


# In[27]:

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

