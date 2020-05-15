# learn_001.py
# 学习tf.placeholder的用法，该方法作为一个占位符，相当于函数中的形参，用于定义tensorflow计算
# 图中的相关参数。
import tensorflow as tf
import numpy as np
import os

x = tf.placeholder(tf.float32, [5, 5], name='x')
y = tf.placeholder(tf.float32, [5, 5], name='y')
z = tf.matmul(x, y, name='matmul')

with tf.Session() as sess:
    x_value = np.random.rand(5, 5)
    y_value = np.random.rand(5, 5)
    z_value = sess.run(z, feed_dict={x: x_value, y: y_value})
    print(x_value)
    print(y_value)
    print(z_value)
