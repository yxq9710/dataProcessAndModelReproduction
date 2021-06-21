from tensorflow.keras import Model, losses, optimizers, regularizers
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
import tensorflow as tf

from codes.Dcnn_add_Dcnn.function_test import change_train_data


class testModel(Model):
    def __init__(self, vocab_size, output_dim, embedding_dim, max_len, embedding_matrix=None, attn_mask=False):
        super(testModel, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_states = 64
        self.l2 = 0.1

        self.dw1 = Dense(embedding_dim)
        self.dw2 = Dense(embedding_dim)
        self.du1 = Dense(embedding_dim)
        self.du2 = Dense(embedding_dim)
        self.word_embed = Embedding(vocab_size + 1, embedding_dim, input_length=max_len)
        self.position_embed = Embedding(max_len, embedding_dim, input_length=max_len)

        self.drop = Dropout(rate=0.3)
        self.d4 = Dense(embedding_dim)  # 后接dropout和tf.nn.relu
        self.flat = Flatten()
        self.d5 = Dense(output_dim, use_bias=False, kernel_regularizer=regularizers.l2(self.l2))

    def call(self, inputs):
        sentence_embedding = self.attention(inputs)
        output = self.FCLayer(sentence_embedding)
        return output

    def attention(self, inputs):
        """
          1. 获取输入
          2. 获得word和position_embedding  =====>  (None, max_len, embedding_dim)
            2.1 将embedding新增一维         =====>  (None, max_len, 1, embedding_dim) 和 (None, 1, max_len, embedding_dim)
          3. 获得word_specific矩阵S_i (共有max_len个)
          4. 获取weight_i矩阵 共有(max_len个)
          5. 每个S_i和weight_i进行加权得到一个word_i, 将word_i进行堆叠即得到了句子的embedding表示 (batches, max_len, embedding_dim)

        """
        word_data, position_data, sentence_length = inputs

        word_embedding = self.word_embed(word_data)
        position_embedding = self.position_embed(position_data)
        word_embedding = word_embedding + position_embedding

        word_embedding1 = tf.expand_dims(word_embedding, axis=2)  # (None, 64, 1, 300)
        word_embedding1 = tf.tile(word_embedding1, [1, 1, self.max_len, 1])  # (None,64,64, 300)
        word_embedding2 = tf.expand_dims(word_embedding, axis=1)  # (None, 1, 64, 300)  ==> word_specific
        # word_embedding2 = tf.tile(word_embedding2, [1, ])
        S_i = self.dw1(word_embedding1) + self.dw2(word_embedding2)  # (None, 64, 64, 300) --> axis=1里面包含了64个不同的S_i
        S_i = tf.nn.relu(S_i)
        weight_i = tf.nn.tanh(self.du1(word_embedding1) + self.du2(S_i))  # (None, 64, 64, 300)
        sentence_embedding = tf.reduce_sum(weight_i * S_i, axis=2)  # (None, 64, 300)
        # weight_i = tf.nn.tanh(self.du1(word_embedding2) + self.du2(S_i))  # (None, 64, 64, 300)
        # sentence_embedding = tf.reduce_sum(weight_i * word_embedding2, axis=2)  # (None, 64, 300)

        return sentence_embedding, weight_i

    def FCLayer(self, attn_result):
        attn_result = self.d4(attn_result)
        attn_result = self.drop(attn_result)
        attn_result = tf.nn.relu(attn_result)
        # attn_result = self.selu(attn_result)
        attn_result = self.flat(attn_result)
        out = self.d5(attn_result)
        return out


# model = testModel(vocab_size=16789, output_dim=1, embedding_dim=200, max_len=35)
# model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
#               loss=losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# import numpy as np
#
# train = np.load("train_array.npy")
# x_train = train[:, 1: -1]
# y_train = train[:, 0]
#
# test = np.load("test_array.npy")
# x_test = test[:, 1: -1]
# y_test = test[:, 0]
# x_dev = x_test
# y_dev = y_test
# x_train, x_dev, x_test = change_train_data(x_train, x_dev, x_test, 35)
# history = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=(x_dev, y_dev), verbose=1)
