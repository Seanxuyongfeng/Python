
# coding: utf-8

# In[11]:
#conda install pillow

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def ImgConvolve(image_array, kernel):
    '''
    image_array: 原灰度图像矩阵
    kernel: 卷积核
    返回值: 原图像和卷积核运算后的结果
    '''
    image_arr = image_array.copy()
    img_width,img_height = image_arr.shape
    k_width,k_height = kernel.shape
    #根据原图像和kernel，计算padding的大小
    # (img_width - (img_width - k_width + 1))/2 = (k_width - 1)/2
    padding_width = int((k_width - 1)/2)
    padding_height = int((k_height - 1)/2)
    
    #创建一个临时存储空间，0初始化，其中 + 后面的是属于padding的内容,
    #两边都有padding
    temp = np.zeros([img_width + padding_width*2, img_height + padding_height*2])
    #将原图像拷贝到临时图片的中央
    temp[padding_width:padding_width+img_width, padding_height:padding_height+img_height] = image_arr[:,:]
    
    #创建输出图像的存储空间
    output = np.zeros_like(a=temp)
    
    for i in range(padding_width, padding_width + img_width):
        for j in range(padding_height, padding_height + img_height):
            output[i][j] = int(np.sum(temp[i-padding_width:i+padding_width+1,j-padding_width:j+padding_width+1]*kernel))
    
    return output[padding_width:padding_width+img_width,padding_height:padding_height+img_height];



#提取竖直方向特征
kernel_1 = np.array(
                [[-1,0,1],
                [-2,0,2],
                [-1,0,1]])

#提取水平方向特征
kernel_2 = np.array(
                [[-1,-2,-1],
                [0,0,0],
                [-1,-2,-1]])

#Laplace扩展算子
kernel_3 = np.array(
                [[1,1,1],
                [1,-8,1],
                [1,1,1]])

#打开图像并转化为灰度图像
image = Image.open("juanji.jpg").convert("L")

#将图像转化为数组
image_array = np.array(image)

#卷积操作
sobel_x = ImgConvolve(image_array,kernel_1)
sobel_y = ImgConvolve(image_array,kernel_2)
laplace = ImgConvolve(image_array,kernel_3)


#显示图像

plt.imshow(image_array,cmap=cm.gray)
plt.axis("off")
plt.show()

plt.imshow(sobel_x,cmap=cm.gray)
plt.axis("off")
plt.show()

plt.imshow(sobel_y,cmap=cm.gray)
plt.axis("off")
plt.show()

plt.imshow(laplace,cmap=cm.gray)
plt.axis("off")
plt.show()


# In[ ]:





# In[ ]:




