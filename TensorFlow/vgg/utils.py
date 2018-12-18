
# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy as np
import os
from vgg_preprocess import preprocess_for_train

img_width = 224
img_height = 224

#数据输入
def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root,name))
        for name in sub_folders:
            print('subdir:',name)
            temp.append(os.path.join(root,name))
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('/')[-1]
        if letter == 'cat':
            labels = np.append(labels,n_img*[0])
        else:
            labels = np.append(labels,n_img*[1])
    #shuffle
    print('images:',type(images))
    print('len images:',len(images))
    print('labels:',type(labels))
    print('len labels:',len(labels))
    temp = np.array([images,labels])
    temp = temp.transpose()
    print('temp:',type(temp))
    np.random.shuffle(temp)
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]
    
    return image_list, label_list

def get_file_forwindows(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            temp_root = root
            temp_root += '/'
            images.append(os.path.join(temp_root,name))
        for name in sub_folders:
            print('subdir :',name)
            temp.append(os.path.join(root,name))
    labels = []

    for img in images:
        #print('img:',img)
        letter = img.split('/')[-1]
        letter = letter.split('.')[0]
        #print('letter',letter)
        if letter == 'cat':
            labels = np.append(labels,[0])
            print("cat")
        else:
            labels = np.append(labels,[1])
            print("dog")

    temp = np.array([images,labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]
    
    return image_list, label_list

def get_batch(image_list, label_list,img_width,img_height,batch_size,capacity):
    image = tf.cast(image_list,tf.string)
    label = tf.cast(label_list, tf.int32)
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    
    image = tf.image.decode_jpeg(image_contents,channels=3)
    image = preprocess_for_train(image,224,224)
    image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity)
    label_batch = tf.reshape(label_batch,[batch_size])
    
    return image_batch,label_batch

def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample,n_class))
    onehot_labels[np.arange(n_sample),labels] = 1
    return onehot_labels
