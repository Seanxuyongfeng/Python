
# coding: utf-8

# In[5]:


# 在Jupyter中，使用matplotlib显示图像需要设置 inline 模式，否则不会显示图像, conda install matplotlib
get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt # 载入matplotlib
import numpy as np #载入 numpy
import tensorflow as tf

#设置随机种子
np.random.seed(5)

# 直接采用 np 生成等差数列的方法，生成100个点，每个点的取值在 -1 ，1 之间
x_data = np.linspace(-1,1,100)

# y = 2x + 1 + 噪声, 其中噪声的维度与x_data一致
y_data = 2 * x_data + 1.0 + np.random.randn(*x_data.shape) * 0.4

#画出随机生成的数据的散点图
plt.scatter(x_data, y_data)

#画出我们想要学习到的线性函数 y = 2x + 1
plt.plot(x_data, 2*x_data + 1.0, color = 'red', linewidth=3)




###################定义模型##############################

# x是特征值，y是标签值
x = tf.placeholder("float", name = "x")
y = tf.placeholder("float", name = "y")

w = tf.Variable(1.0, name="w0")
b = tf.Variable(0.0, name="b0")

def model(x, w, b):
    return tf.multiply(x, w) + b

#pred是预测值，前向计算
pred = model(x, w, b)

###########训练模型#########

#设置 迭代次数(训练轮数)
train_epochs = 10

# 学习率
learning_rate = 0.05

#
display_step = 10

#采用 均方差作为损失函数
loss_function = tf.reduce_mean(tf.square(y - pred)) # y是真实的标签值

#梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)


#############初始化变量############
sess = tf.Session()


init = tf.global_variables_initializer()

sess.run(init)

##############迭代训练###################
# 轮数为train_epochs, 采用SGD随机梯度下降优化方法

step = 0 #记录训练步数
loss_list = [] #用于保存loos值的列表

for epoch in range(train_epochs):
    for xs, ys in zip(x_data, y_data):
        _, loss = sess.run([optimizer, loss_function], feed_dict={x: xs, y:ys})
        loss_list.append(loss)
        step = step + 1
        if step % display_step == 0:
            print("Train Epoch:",'%02d' % (epoch+1), "Step: %03d" % (step), "loss",                  "{:.9f}".format(loss))

    #sess.run(optimizer,feed_dict={x: x_data, y:y_data})
    
    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    plt.plot(x_data,w0temp*x_data + b0temp)


print("w:", sess.run(w))
print("b:", sess.run(b))

# loss 结果可视化
#plt.plot(loss_list)
#plt.plot(loss_list,'r+')
#loss中大于1的打印出来
print([x for x in loss_list if x>1])
#####结果可视化###############

#plt.scatter(x_data,y_data,label='Original data')
#plt.plot(x_data, x_data * sess.run(w) + sess.run(b), \
#        label='Fitted line',color='r',linewidth=3)
#plt.legend(loc=2)#通过参数loc指定图例位置


#####################进行预测###########
x_test = 3.43
#pred 就是之前定义的模型
predict = sess.run(pred, feed_dict={x: x_test})
print("预测值: %f" % predict)


# In[ ]:




