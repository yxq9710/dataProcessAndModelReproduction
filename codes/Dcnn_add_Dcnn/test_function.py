""""
matmul 与 * 的区别
result:
    1) matmul时矩阵乘法，按照 行 X 列 的方式进行(vector-wise)， "*" 是element-wise上的操作, multiply与 * 相同
    1) python 按照维度来说是： 先列后行， 即axis=0表示的是列
    2) python在进行矩阵+或者*时， 可以同时在多个维度上进行广播
"""

import numpy as np
import tensorflow as tf
# import nltk

a = tf.random.normal(shape=[100,64,300])
b = tf.transpose(a, perm=[0, 2, 1])
c = tf.matmul(a, b)
print(c.shape)

import sklearn
print(sklearn.__version__)

a = None
assert a is None
b = 3
print(b)

b = tf.constant([3., 2., 5, 4, 6, 7, 1, 8, 9, 10], shape=(1, 10))
a = tf.constant([1., 2, 3, 4, 5, 6, 7, 8, 9, 10], shape=(1, 10))
print(a)
print(b)
a = tf.nn.dropout(a, rate=0.7)
b = tf.nn.dropout(b, rate=0.7)
print(a)
print(b)

a = tf.random.normal(shape=(1, 2))
b = tf.random.normal(shape=(1, 2))
print(a*b)
print(tf.matmul(a, tf.transpose(b)))

a = tf.random.normal(shape=(20, 3, 5, 1))
b = tf.random.normal(shape=(20, 3, 1, 35))
c = tf.matmul(a, b)
print(c.shape)

a = tf.constant([1, 2, 3], shape=(1, 3))
b = tf.constant([4, 5], shape=(2, 1))
print(a*b)

a = tf.constant([1,2,3,4,5,6], shape=(2,3))
print(a)
b = tf.expand_dims(a, axis=1)
print(b)


a = tf.constant([1,1,2,2],shape=(2,2))
print(a)
a = tf.reduce_sum(a, axis=0)
print(a)
a = a[..., tf.newaxis]
a = tf.tile(a, [1, 1, 3])
print(a)


# a = np.array([1, 2, 3, 4])
m = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
a = tf.reshape(m, [-1, 2, 3])
b = tf.reshape(m, [-1, 3, 2])
c = tf.reshape(m, [2, -1, 3])
result = tf.equal(a, c)

# a = a[..., tf.newaxis]
# print(a)
b = tf.constant([1, 2], shape=(2, 1))
c = a * b
print(c)

d = tf.matmul(tf.transpose(a), b)
print(d)

print("测试tf.tile")
m = tf.constant([1, 2, 3, 4, 5, 6], shape=(2, 3, 1))
print(m)
n = tf.tile(m, multiples=[1, 1, 3])
print(n)
