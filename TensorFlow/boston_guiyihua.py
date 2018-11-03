
# coding: utf-8

# In[31]:


get_ipython().magic('matplotlib notebook')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd #conda install pandas
from sklearn.utils import shuffle #conda install scikit-learn

df = pd.read_csv("boston.csv",header=0)

df = df.values

df = np.array(df)


#对样本数据，前12列做归一化处理，针对整列数据进行归一化
for i in range(12):
    df[:,i]=df[:,i]/(df[:,i].max() - df[:,i].min())

# 获取前12列，归一化之后逇数据
x_data = df[:,:12]

#最后一列为标签数据
y_data = df[:,12]
#print(y_data,y_data.shape)


#定义 存放输入值的地方，12列的数据
#在这里其实tf.placeholder(tf.float32,[1,12],name = "X") 也行，就是一行
x = tf.placeholder(tf.float32,[None,12],name = "X")

#定义 存放输入标签的地方，一列数据,也就是第十三列
#在这里 tf.placeholder(tf.float32,[1,1],name="Y") 也行，就是一个标量
y = tf.placeholder(tf.float32,[None,1],name="Y")

#定义命名空间
with tf.name_scope("Model"):
    
    #定义权重，因为有12个输入因素，上面定义的placeholder,所以有12个权重
    w = tf.Variable(tf.random_normal([12,1],stddev=0.01), name="W")
    
    #定义标量
    b = tf.Variable(1.0,name="b")
    
    def model(x,w,b):
        #这里tf.matmul(x,w) 是一个标量，因此，x表示其实是数据集里面的一行
        # y = x1*w1 + x2*w2 + ... + x12 * w12 + b
        return tf.matmul(x,w) + b
    
    #定义计算操作，前向计算节点
    pred = model(x,w,b)

# 开始训练模型


#迭代轮次
train_epochs = 50

#学习率
learning_rate = 0.01

#定义损失函数
with tf.name_scope("LossFunction"):
    loss_function = tf.reduce_mean(tf.pow(y-pred, 2))

#创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

#迭代训练
for epoch in range(train_epochs):
    loss_sum = 0.0
    #这里由于x_data有506行，循环506次，这是所有的样本运行了一次
    for xs, ys in zip(x_data, y_data):
            #reshape是将xs，ys转化为二维数组，因为Feed数据必须和Placeholder的shapce一致
            xs = xs.reshape(1,12)
            ys = ys.reshape(1,1)
            
            _, loss = sess.run([optimizer, loss_function],feed_dict={x:xs,y:ys})
            
            loss_sum = loss_sum + loss
        
    #经过全样本一轮训练之后，打乱样本数据，为了进行下一轮训练,打乱是为了训练更加合理
    xvalues, yvalues = shuffle(x_data, y_data)
    
    b0temp=b.eval(session=sess)
    w0temp=w.eval(session=sess)
    loss_average = loss_sum/len(y_data)
    
    #打印训练一轮的结果
    print("epoch=",epoch+1,"loss=",loss_average,"b=",b0temp,"w=",w0temp)

