# %%
"""
    step：
        1.选择数据集并导入数据
        2.数据预处理：1）使用uni-gram和bi-grams对数据集进行Token(特征化)
                    2）特征向量化(此处使用one-hot encoding)
        3.训练模型：根据输入样本得到具有各概率值的分类器
        4.测试模型：分类
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_data():
    imdb = tf.keras.datasets.imdb
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
    # print(len(x_train))
    # print(x_train[0])
    # print(len(x_train[1]))
    # print(len(x_train[0]))
    return x_train, y_train, x_test, y_test


# %%
def data_process(x_train, x_test):
    """
    step:
        1.将样本数据填充为fixed length（one-hot时为词表长度）
        2.切片进行token话
    """
    # print(x_train.shape)
    # x_train = np.minimum(np.array(x_train), np.ones_like(x_train) * 255)
    vocab_size = 5000
    # vocab_size = 256
    train_length = len(x_train)
    train_data = np.zeros((train_length, vocab_size))
    for i in range(len(x_train)):
        train_data[i, x_train[i]] = 1
    # print(train_data.shape)

    test_length = len(x_test)
    test_data = np.zeros((test_length, vocab_size))
    for i in range(len(x_test)):
        test_data[i, x_test[i]] = 1
    # print(test_data.shape)
    return train_data, test_data


# %%
def NB_train(train_data, y_train):
    """
    数据来源：训练集数据
    step:
        1.计算P(c=C0), P(c=C1)
        2.计算P(x1|C0), P(x2|C0), ... ,P(xn|C0)
        3.计算P(x1|C1), P(x2|C1), ... ,P(xn|C1)
    """
    train_length = train_data.shape[0]  # 样本数
    sum_C0 = 0  # C0类样本的数目
    sum_C1 = 0
    for i in range(len(y_train)):
        if y_train[i] == 0:
            sum_C0 += 1
        else:
            sum_C1 += 1
    P0 = sum_C0 / train_length  # 属于C0类的概率
    P1 = sum_C1 / train_length

    n = train_data.shape[1]
    sum_xn_0 = np.zeros((1, n))
    sum_xn_1 = np.zeros_like(sum_xn_0)
    for i in range(0, train_length):
        for j in range(0, n):
            if train_data[i, j] == 1:
                if y_train[i] == 0:
                    sum_xn_0[0, j] += 1
                elif y_train[i] == 1:
                    sum_xn_1[0, j] += 1
    px0 = sum_xn_0 / sum_C0
    px1 = sum_xn_1 / sum_C1
    print(px0.shape)
    print(px1.shape)
    print(px0.shape[1])

    return P0, P1, px0, px1


# %%
def NB_test(test_data, y_test, P0, P1, px0, px1):
    """
    通过训练得到的分类器进行预测
    数据来源：测试集数据，训练模型
    step:
        1.遍历每个样本
        2.计算其属于C0或C1的概率值fc0, fc1
        3.比较fc0和fc1，将结果放入predict
        4.将predict与真实label比较，得到准确率
    """
    m, n = test_data.shape
    print(m, n)
    predict = np.zeros((m,))  # 比较fc0和fc1
    for i in range(m):
        fc0 = P0  # 属于第一类的概率值
        fc1 = P1
        for j in range(0, n):
            if test_data[i, j] == 1:
                fc0 *= px0[0, j]
                fc1 *= px1[0, j]
        if fc0 >= fc1:
            predict[i] = 0
        else:
            predict[i] = 1
        # print("fc0: {}, fc1: {}".format(fc0, fc1))

    acc = np.sum((predict == y_test)) / m
    print(acc)
    return acc


x_train, y_train, x_test, y_test = load_data()
print(y_train.shape)
train_data, test_data = data_process(x_train, x_test)
P0, P1, px0, px1 = NB_train(train_data, y_train)
acc = NB_test(test_data, y_test, P0, P1, px0, px1)

# %%
