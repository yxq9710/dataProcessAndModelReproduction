"""
Conv1作用于embedding_dim层面， Conv2作用于word层面，
results:
    Conv1: 0.82
    Conv2: 0.83              (epoch=2时提前终止了)
    Conv1 + Conv2: 0.80
    Conv2 + Conv1: 0.8276    (epoch = 2时取得， 最优情况在epoch=2到epoch=3之间)， --运行了一次,才0.19 (波动性可真大)
可以用keras.utils.plot_model()画模型图（只能画def和Sequential的  ==》 class有点不受欢迎）
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Conv2D, MaxPooling2D, \
    Dense, Concatenate, Flatten, Dropout, BatchNormalization, ReLU
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import plot_model


class CNN_CNN(Model):
    def __init__(self, vocab_size=10000, embedding_dim=100, batchsz=50, max_len=64):
        super(CNN_CNN, self).__init__()
        self.embedding = Embedding(vocab_size + 1, embedding_dim, input_length=max_len)
        self.conv11 = Conv2D(100, kernel_size=(1, 3), padding="valid",
                             activation='relu', kernel_constraint=max_norm(2, axis=[0, 1, 2]))  # axis=3的维度为channel
        self.conv12 = Conv2D(100, kernel_size=(1, 4), padding="valid",                          # axis=2的维度为embedding_dim
                             activation='relu', kernel_constraint=max_norm(2, axis=[0, 1, 2]))  # axis=1的维度为word
        self.conv13 = Conv2D(100, kernel_size=(1, 5), padding="valid",                          # axis=0的维度为sentence
                             activation='relu', kernel_constraint=max_norm(2, axis=[0, 1, 2]))
        # self.batch_norm = BatchNormalization()    # 很耗时间
        # self.relu = ReLU()
        self.pool11 = MaxPooling2D(pool_size=(1, embedding_dim - 3 + 1))        # strides = pool_size(default)
        self.pool12 = MaxPooling2D(pool_size=(1, embedding_dim - 4 + 1))
        self.pool13 = MaxPooling2D(pool_size=(1, embedding_dim - 5 + 1))
        self.conv21 = Conv2D(100, kernel_size=(3, 1), padding="same",
                             activation='relu', kernel_constraint=max_norm(2, axis=[0, 1, 2]))
        self.conv22 = Conv2D(100, kernel_size=(4, 1), padding="same",
                             activation='relu', kernel_constraint=max_norm(2, axis=[0, 1, 2]))
        self.conv23 = Conv2D(100, kernel_size=(5, 1), padding="same",
                             activation='relu', kernel_constraint=max_norm(2, axis=[0, 1, 2]))
        self.pool21 = MaxPooling2D(pool_size=(max_len - 3 + 1, 1))
        self.pool22 = MaxPooling2D(pool_size=(max_len - 4 + 1, 1))
        self.pool23 = MaxPooling2D(pool_size=(max_len - 5 + 1, 1))
        self.drop = Dropout(rate=0.5)
        self.flatten = Flatten()
        self.dense1 = Dense(32)
        self.dense2 = Dense(1, activation="sigmoid")

    def call(self, inputs):
        x_input = inputs
        embed = self.embedding(x_input)
        embed = embed[..., tf.newaxis]    # channel last(default)

        x = embed
        conv21 = self.conv21(x)
        conv22 = self.conv22(x)
        conv23 = self.conv23(x)
        pool21 = self.pool21(conv21)
        pool22 = self.pool22(conv22)
        pool23 = self.pool23(conv23)
        x = Concatenate(axis=1)([pool21, pool22, pool23])

        embed = x
        conv11 = self.conv11(embed)
        conv12 = self.conv12(embed)
        conv13 = self.conv13(embed)
        pool11 = self.pool11(conv11)
        pool12 = self.pool12(conv12)
        pool13 = self.pool13(conv13)
        x = Concatenate(axis=2)([pool11, pool12, pool13])

        flatten = self.flatten(x)
        dense1 = self.dense1(flatten)
        drop = self.drop(dense1)
        output = self.dense2(drop)

        return output


if __name__ == '__main__':
    batchsz = 50
    max_len = 64
    model = CNN_CNN()
    model.build(input_shape=(batchsz, max_len))
    plot_model(model, show_shapes=True)
    model.summary()

