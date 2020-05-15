# learn_002.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import minist_data_read as mdRead


def conv2d(x, weights):
    # x:指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
    # 具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度,图像通道数]，注意这是一个4维的Tensor，要求类型为
    # float32和float64其中之一
    # weights:相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width,
    # in_channels, out_channels] 这样的shape
    # strides: 卷积时在图像每一维的步长，这是一个一维的向量，长度4
    # padding: string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）就是VALID只能匹配内部像
    # 素；而SAME可以在图像外部补0,从而做到只要图像中的一个像素就可以和卷积核做卷积操作,而VALID不行
    return tf.nn.conv2d(x, weights, strides=[1, 2, 2, 1], padding='SAME')


train_images = mdRead.load_train_images()
weights0 = np.array([[[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]],
                     [[1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]],
                     [[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]]
                     ])
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        t = tf.convert_to_tensor(train_images[i], tf.float32, name='t')
        t = tf.reshape(t, [1, 28, 28, 1])
        weights2 = tf.convert_to_tensor(weights0, tf.float32, name='t')
        weights2 = tf.reshape(weights2, [3, 3, 1, 3])
        image2 = tf.nn.conv2d(t, weights2, strides=[1, 1, 1, 1], padding='SAME')
        # image2 = sess.run(result, feed_dict={x: image1, weights: weights1})
        # perm-控制转置的操作，perm = [0, 1, 3, 2]表示，把将要转置的第0和第1维度不变，将第2和第3维度进行转置
        # perm-控制转置的操作，perm = [3, 0, 1, 2]表示，把[1,28,28,3]转为[3,1,28,28]
        my_result = sess.run(tf.transpose(image2, [3, 0, 1, 2]))
        print(my_result)
        plt.subplot(2, 2, 1)
        plt.imshow(train_images[i], cmap='gray')
        plt.xlabel('Origin Image%s' % i)
        plt.subplot(2, 2, 2)
        plt.imshow(my_result[0][0], cmap='gray')
        plt.xlabel('Filter1 result%s' % i)
        plt.subplot(2, 2, 3)
        plt.imshow(my_result[1][0], cmap='gray')
        plt.xlabel('Filter2 result%s' % i)
        plt.subplot(2, 2, 4)
        plt.imshow(my_result[2][0], cmap='gray')
        plt.xlabel('Filter3 result%s' % i)
        plt.pause(1)
