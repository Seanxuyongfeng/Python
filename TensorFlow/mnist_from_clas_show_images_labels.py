
# coding: utf-8

# In[39]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pylab as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# mnist 中每张图片共有28*28个像素点
x = tf.placeholder(tf.float32, [None, 784], name="X")

# y是分类结果，一个结果又10个元素，代码是每个数字的概率
y = tf.placeholder(tf.float32, [None,10], name="Y")

#定义变量,W 随机产生一些值，b指定0
W = tf.Variable(tf.random_normal([784, 10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")

forward = tf.matmul(x,W) + b


#将结果forward进行分类化,总概率为1
pred = tf.nn.softmax(forward)

###
train_epochs = 50 #训练轮数
batch_size = 100 #每次训练样本数
total_batch = int(mnist.train.num_examples/batch_size) #一轮训练多少次
display_step = 1 #显示粒度
learning_rate = 0.01


#定义交叉熵损失函数,在这里 pred和y都不是一个，而是一批，所以这里取了这一批的均值reduce_mean作为损失值
loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 检查预测类别tf.argmax(pred,1) 与实际类别tf.argmax(y,1) 的匹配情况
#这里也是批量的准确率correction_prediction，里面的值就是pred中的下标值，就是0-9之间的
#因为argmax 会取出pred每个中的最大值，也就是分出来的类别是什么
#相等为true，不相等为false
correction_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))


#将结果转换为float32类型，因为之前的结果是true，和false是无法直接进行运算的
#求这些prediction的平均值，作为准确率，也就是批量的预测的均值
#correct_prediction 转换之后，内容不是0就是1，就可以进行计算了
accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)


#开始训练
for epoch in range(train_epochs):
    
    #全部样本训练一轮
    for batch in range(total_batch):
        xs, ys = mnist.train.next_batch(batch_size) #读取批次数据
        sess.run(optimizer,feed_dict={x:xs,y:ys})
    
    #一轮训练完毕,使用验证数据计算误差和准确率
    loss, acc = sess.run([loss_function, accuracy],
                       feed_dict={x:mnist.validation.images,y:mnist.validation.labels})
    
    #打印训练过程中的信息
    if (epoch+1) % display_step == 0:
        print("Train Epoch:", '%02d' %(epoch+1), "Loss=",'{:.9f}'.format(loss),              "Accuracy=","{:.4f}".format(acc))
print("Train Finished!")

##############训练完成之后，在测试集上评估模型的准确率

accu_test = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})

print("Test Accuracy:", accu_test)


################在建立模型并进行训练之后，若认为准确率可以接收，则可以使用此模型进行预测

prediction_result = sess.run(tf.argmax(pred, 1), feed_dict={x:mnist.test.images})

print(prediction_result[0:10])


#########################可视化
def plot_images_labels_prediction(images,     #图像列表
                                  labels,     #标签列表
                                  prediction, #预测值列表
                                  index,      #从第index个显示显示
                                  num=10):    #缺省一次显示10幅
    fig = plt.gcf() #获取当前图表，Get Current Figure
    fig.set_size_inches(10,12)# 英寸，1英寸等于2.54cm
    if num > 25:
        num = 25      #最多显示25个子图
    
    for i in range(0, num):
        ax = plt.subplot(5,5, i+1) #获取当前要处理的子图
        ax.imshow(np.reshape(images[index],(28,28)), cmap='binary')  #显示第index个图像
        
        title = "label=" + str(np.argmax(labels[index]))  #构建该图上要显示的title
        if len(prediction) > 0:
            title += ".predict=" + str(prediction[index])
        
        ax.set_title(title, fontsize=10) # 显示图上的title信息
        ax.set_xticks([]) #不显示坐标轴
        ax.set_yticks([])
        
        index += 1
    
    plt.show()

plot_images_labels_prediction(mnist.test.images,
                             mnist.test.labels,
                             prediction_result,0,15)


# In[ ]:




