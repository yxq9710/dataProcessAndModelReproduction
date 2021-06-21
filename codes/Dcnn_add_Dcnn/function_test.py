"""
本代码旨在使用textcnn模型测试plot_model的功能和使用方法
result:
    plot_model可以绘制用function定义的model，但不能绘制用class定义的model
    当最后一个Dense(activation='sigmoid)，则不要令losses.BinaryCrossentropy(from_logits=True)

    基础模型比较：
        1—D的CNN比simple2—D的CNN易于调参，目前来说1-D的效果好一些, embedding_dim==300
        LSTM的input为3—D tensor


对DL中维度的理解：
    examples: x.shape = (bs, sl, embedding_dim, channels)
    则在其中channels维度下才涉及到一个个的dada（单个元素）。 
        而每一个embedding_dim都是一系列data的总结
        每一个sl都是一系列embedding_dim的总结
        每一个bs都是一系列sl的总结 
            尤其是当embedding_dim, sl, bs等都只有一个时， 则 ‘一系列’ 代表 ‘所有’， 若tile则tile所有

"""
# %%
import sys

sys.path.append('e:\\myWorks')
# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Embedding, concatenate, Flatten
from tensorflow.keras import Input, optimizers, losses, regularizers, Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from dataset.imdb import data_process_imdb as dp
from dataset.MR.rt_polaritydata import data_process_mr as dp_mr
from codes.Dcnn_add_Dcnn.plt_loss import plot_loss as pl
from codes.Dcnn_add_Dcnn.data_process_SST_2 import get_all_data
# from codes.Dcnn_add_Dcnn.utils import convert_to_onehot
from tensorflow.keras.utils import to_categorical as convert_to_onehot


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):  # 设置epoch从0到4000时lr的变化函数, warmup_steps越大，最大的lr越小
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(300, 100)  # 学习率的自变量为step，且不同epoch间的step会累加
LARGE_MASK = 1e30


def TextCNN(vocab_size, output_dim, embedding_dim, max_len, embedding_matrix=None):
    x_input = Input(shape=(max_len,))
    if embedding_matrix is None:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len)(x_input)
    else:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len,
                      weights=[embedding_matrix], trainable=False)(x_input)
    x = x[..., tf.newaxis]
    filters = [100, 100, 100]
    output_pool = []
    kernel_sizes = [3, 4, 5]
    for i, kernel_size in enumerate(kernel_sizes):
        conv = Conv2D(filters=filters[i], kernel_size=(max_len, embedding_dim),
                      padding='valid', kernel_constraint=max_norm(3, [0, 1, 2]))(x)
        pool = MaxPool2D(pool_size=(max_len - kernel_size + 1, 1))(conv)
        # pool = tf.keras.layers.GlobalAveragePooling2D()(conv)  # 1_max pooling
        output_pool.append(pool)
        # logging.info("kernel_size: {}, conv.shape: {}, pool.shape: {}".format(kernel_size, conv.shape, pool.shape))
        print("kernel_size: {}, conv.shape: {}, pool.shape: {}".format(kernel_size, conv.shape, pool.shape))
    output_pool = concatenate([p for p in output_pool])
    # logging.info("output_pool.shape: {}".format(output_pool.shape))
    print("output_pool.shape: {}".format(output_pool.shape))

    x = Dropout(rate=0.5)(output_pool)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    y = Dense(output_dim, activation='sigmoid')(x)
    # y = Dense(output_dim, activation='softmax')(x)
    model = tf.keras.Model([x_input], y)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss=losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
    #               loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    #               metrics=['accuracy'])
    model.summary()
    return model


def _1D_CNN(vocab_size, output_dim, embedding_dim, max_len, embedding_matrix=None):
    x_input = Input(shape=(max_len,))
    if embedding_matrix is None:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len)(x_input)
    else:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len,
                      weights=[embedding_matrix], trainable=False)(x_input)
    # x = x[..., tf.newaxis]
    x = tf.keras.layers.Conv1D(filters=100, kernel_size=3, strides=1)(x)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(16)(x)
    y = Dense(output_dim, activation='sigmoid')(x)
    model = tf.keras.Model([x_input], y)
    model.compile(optimizer=optimizers.Adam(lr=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def _simple2D_CNN(vocab_size, output_dim, embedding_dim, max_len, embedding_matrix=None):
    x_input = Input(shape=(max_len,))
    if embedding_matrix is None:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len)(x_input)
    else:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len,
                      weights=[embedding_matrix], trainable=False)(x_input)
    x = x[..., tf.newaxis]
    x = tf.keras.layers.Conv2D(filters=100, kernel_size=(3, embedding_dim), strides=1)(x)
    x = tf.keras.layers.GlobalMaxPool2D()(x)
    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(16)(x)
    y = Dense(output_dim, activation='sigmoid')(x)
    model = tf.keras.Model([x_input], y)
    model.compile(optimizer=optimizers.Adam(lr=0.0003),
                  loss=losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    model.summary()
    return model


def _RNN(vocab_size, output_dim, embedding_dim, max_len, embedding_matrix=None):
    x_input = Input(shape=(max_len,))
    if embedding_matrix is None:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len)(x_input)
    else:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len,
                      weights=[embedding_matrix], trainable=False)(x_input)
    x = tf.keras.layers.SimpleRNN(64)(x)  # RNN适用于堆砌多层SimpleRNN的 --> 所以参数为cell
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    # x = Dropout(rate=0.5)(x)
    y = Dense(output_dim, activation='sigmoid')(x)
    model = tf.keras.Model([x_input], y)
    model.compile(optimizer=optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def _LSTM(vocab_size, output_dim, embedding_dim, max_len, embedding_matrix=None):
    x_input = Input(shape=(max_len,))
    if embedding_matrix is None:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len)(x_input)
    else:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len,
                      weights=[embedding_matrix], trainable=False)(x_input)
    x = tf.keras.layers.LSTM(64)(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    # x = Dropout(rate=0.5)(x)
    y = Dense(output_dim, activation='sigmoid')(x)
    model = tf.keras.Model([x_input], y)
    model.compile(optimizer=optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def only_attention(vocab_size, output_dim, embedding_dim, max_len, embedding_matrix=None):
    x_input = Input(shape=(max_len,))
    if embedding_matrix is None:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len)(x_input)
    else:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len,
                      weights=[embedding_matrix], trainable=False)(x_input)

    # ========================== target-attention ============================
    sentence_embedding = tf.reduce_mean(x, axis=1, keepdims=True)  # (10001,1,100)
    w1 = Dense(64)(x)  # key vector
    w2 = Dense(64)(sentence_embedding)  # query vector
    score = Dense(1)(tf.nn.tanh(w1 + w2))  # x 为 value vector
    weights = tf.nn.softmax(score, axis=1)  # (None, 64, 1)
    context_embedding = tf.reduce_sum(weights * x, axis=1, keepdims=True)  # (None, 1, 100)

    context_embedding = tf.tile(context_embedding, [1, max_len, 1])

    x = concatenate([x, context_embedding], axis=-1)  # (10001,65,100)是错误的，65意味着有65个单词 --> 应该为(10001,150,600)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    y = Dense(output_dim, activation='sigmoid')(x)
    model = tf.keras.Model([x_input], y)
    model.compile(optimizer=optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def self_attention(vocab_size, output_dim, embedding_dim, max_len, embedding_matrix=None, attn_mask=None):
    x_input = Input(shape=(max_len,))
    if embedding_matrix is None:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len)(x_input)
    else:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len,
                      weights=[embedding_matrix], trainable=False)(x_input)

    # =============================== self-attention ===============================
    w1 = Dense(64)(x)  # query vector  --在value vector的dimension_dim上进行映射
    w2 = Dense(64)(x)  # key vector
    w1 = tf.expand_dims(w1, axis=2)  # (None, 64, 1, 16)
    w2 = tf.expand_dims(w2, axis=1)  # (None, 1, 64, 16)
    score = Dense(embedding_dim)(tf.nn.tanh(w1 + w2))  # 用1替换embedding_dim
    weights = tf.nn.softmax(score, axis=2)
    x = tf.expand_dims(x, axis=1)
    context_embedding = tf.reduce_sum(weights * x, axis=2)
    x = Flatten()(context_embedding)

    x = Dense(16, activation='relu')(x)  # use_bias = True(default)
    # x = Dropout(rate=0.1)(x)
    # y = Dense(output_dim, activation='sigmoid',
    #           kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.001))(x)
    y = Dense(output_dim, activation='sigmoid')(x)
    model = tf.keras.Model([x_input], y)
    model1 = tf.keras.Model([x_input], weights)
    model.compile(optimizer=optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model, model1


def self_attention_mask(vocab_size, output_dim, embedding_dim, max_len, embedding_matrix=None, attn_mask=False):
    x_input = Input(shape=(max_len,))
    # a = x_input[0]
    # attn_mask = x_input[1]
    # x_input = a

    if embedding_matrix is None:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len, mask_zero=True)(x_input)
    else:
        x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len, mask_zero=True,
                      weights=[embedding_matrix], trainable=False)(x_input)

    # =============================== self-attention ===============================
    sl = tf.range(max_len)  # [0, 1, 2, ..., 63]   shape=(64, )
    word_col, word_row = tf.meshgrid(sl, sl)  # 刚好构成了word位置的索引矩阵

    direct_mask_1 = tf.greater(word_row, word_col)
    direct_mask_1 = 1 - tf.cast(direct_mask_1, tf.float32)

    direct_mask_2 = tf.greater(word_col, word_row)
    direct_mask_2 = 1 - tf.cast(direct_mask_2, tf.float32)

    direct_mask_1 = pad_axis(direct_mask_1)
    direct_mask_2 = pad_axis(direct_mask_2)

    position_input = sl
    position_embedding = Embedding(input_dim=max_len, output_dim=embedding_dim, input_length=max_len,
                                   name="position_encoding")(
        position_input)  # 随机初始化位置信息
    x = x + position_embedding

    bias = tf.Variable(tf.zeros(shape=(embedding_dim,)))
    val = tf.expand_dims(Dense(embedding_dim, use_bias=False)(x), axis=1) + tf.expand_dims(
        Dense(embedding_dim, use_bias=False)(x), axis=2) + bias
    x_score = fx_q(val, 5.)
    if attn_mask is True:
        attn_mask = tf.cast(x_input, bool)  # 此处有问题
    else:
        attn_mask = None
    if attn_mask is not None:
        attn_mask_1 = tf.logical_and(attn_mask, direct_mask_1)
        attn_mask_2 = tf.logical_and(attn_mask, direct_mask_2)
    else:
        attn_mask_1 = direct_mask_1
        attn_mask_2 = direct_mask_2

    x = tf.expand_dims(x, 1)

    x_1 = attention_mask(x_score, attn_mask_1)
    weights_1 = tf.nn.softmax(x_1, axis=2)
    weights_1 = position_mask(weights_1, attn_mask_1)  # 把填充的为值全部变为0
    context_embedding_1 = tf.reduce_sum(weights_1 * x, axis=2)

    x_2 = attention_mask(x_score, attn_mask_2)
    weights_2 = tf.nn.softmax(x_2, axis=2)
    weights_2 = position_mask(weights_2, attn_mask_2)
    context_embedding_2 = tf.reduce_sum(weights_2 * x, axis=2)

    context_embedding = concatenate([context_embedding_1, context_embedding_2], axis=1)

    x = Flatten()(context_embedding)
    x = Dense(16, activation='relu')(x)  # use_bias = True(default)
    x = Dropout(rate=0.1)(x)
    y = Dense(output_dim, activation='sigmoid')(x)
    # y = Dense(output_dim, activation='sigmoid',
    #           kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.001))(x)
    model = tf.keras.Model([x_input], y)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def pad_axis(direct_mask):
    mask = tf.expand_dims(direct_mask, -1)
    mask = mask[tf.newaxis, ...]
    return mask


def position_mask(embedding, mask):
    """
    参数：mask为三角形的1/0矩阵
    将embedding中forward/backward方向的元素变为0
    """
    mask = tf.multiply(embedding, mask)
    return mask


def attention_mask(embedding, attn_mask):
    """
    参数：attn_mask里面padding的元素全为0
    结果：将padding的值全部变为无穷小
    """
    attn_mask = (1 - tf.cast(attn_mask, tf.float32)) * LARGE_MASK  # 0或者正无穷
    embedding = tf.subtract(embedding, attn_mask)
    return embedding


def fx_q(val, scale=5.):
    return scale * tf.nn.tanh(1. / scale * val)


def lr_(epoch):
    if epoch < 2:
        return 0.0001
    if epoch < 4:
        return 0.001
    if epoch < 6:
        return 0.0005
    if epoch < 10:
        return 0.0001
    else:
        return 5e-5


def mask(pad_data):
    mask_row_numble = []
    for i in range(pad_data.shape[0]):  # i个样本
        count = 0
        for j in range(pad_data[i].shape[0]):
            if pad_data[i][j] != 0:
                count += 1
        mask_row_numble.append(count)
    return np.reshape(np.array(mask_row_numble), [-1, 1])


def process_mask(mask_length, max_len):
    pad_data_tile = np.zeros([mask_length.shape[0], max_len, max_len])
    for i in range(len(mask_length)):
        m = np.squeeze(mask_length[i])
        a = np.ones(shape=(m, m))
        pad_data_tile[i][:m, :m] = a
    return pad_data_tile


def change_train_data(x_train, x_dev, x_test, max_len):
    a = tf.range(max_len)
    b = tf.constant(a, shape=(1, max_len))
    c_train = tf.tile(b, [x_train.shape[0], 1]).numpy()
    c_dev = tf.tile(b, [x_dev.shape[0], 1]).numpy()
    c_test = tf.tile(b, [x_test.shape[0], 1]).numpy()
    # mask_length_train = tf.tile(mask(pad_data=x_train), [1, x_train.shape[1]]).numpy()
    # mask_length_dev = tf.tile(mask(pad_data=x_dev), [1, x_dev.shape[1]]).numpy()
    # mask_length_test = tf.tile(mask(pad_data=x_test), [1, x_test.shape[1]]).numpy()
    mask_length_train = mask(pad_data=x_train)
    mask_length_dev = mask(pad_data=x_dev)
    mask_length_test = mask(pad_data=x_test)

    mask_length_train = process_mask(mask_length_train, max_len)
    mask_length_dev = process_mask(mask_length_dev, max_len)
    mask_length_test = process_mask(mask_length_test, max_len)

    x_train = [x_train, c_train, mask_length_train]
    x_dev = [x_dev, c_dev, mask_length_dev]
    x_test = [x_test, c_test, mask_length_test]
    return x_train, x_dev, x_test


class position_attention(Model):
    def __init__(self, vocab_size, output_dim, embedding_dim, max_len, embedding_matrix=None, attn_mask=False):
        super(position_attention, self).__init__()
        self.attn_mask = 0
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_states = 64
        self.l2 = 0.1
        self.pad_data = 0
        self.word_embedding = 0

        if embedding_matrix is None:
            self.word_embed = Embedding(vocab_size + 1, embedding_dim, input_length=max_len)
        else:
            self.word_embed = Embedding(vocab_size + 1, embedding_dim, input_length=max_len,
                                        weights=[embedding_matrix])
        self.position_embed = Embedding(input_dim=max_len, output_dim=embedding_dim, input_length=max_len)
        self.drop_embedding = Dropout(rate=0.1)
        self.d1 = Dense(embedding_dim)
        self.drop = Dropout(rate=0.3)

        self.d2 = Dense(embedding_dim, use_bias=False)  # 后接dropout=0.7
        self.d3 = Dense(embedding_dim, use_bias=False)
        self.bias = tf.Variable(tf.zeros(shape=(embedding_dim,)))
        self.fusion_bias = tf.Variable(tf.zeros(shape=(embedding_dim,)))
        # 混合attention和输入,后接dropout
        self.d6 = Dense(embedding_dim)
        self.d7 = Dense(embedding_dim)

        self.d1_ex = Dense(embedding_dim)
        self.drop_ex = Dropout(rate=0.3)
        self.d2_ex = Dense(embedding_dim, use_bias=False)  # 后接dropout=0.7
        self.d3_ex = Dense(embedding_dim, use_bias=False)
        self.bias_ex = tf.Variable(tf.zeros(shape=(embedding_dim,)))
        self.fusion_bias_ex = tf.Variable(tf.zeros(shape=(embedding_dim,)))
        self.d6_ex = Dense(embedding_dim)
        self.d7_ex = Dense(embedding_dim)

        self.d1_no = Dense(embedding_dim)
        self.drop_no = Dropout(rate=0.3)
        self.d2_no = Dense(embedding_dim, use_bias=False)  # 后接dropout=0.7
        self.d3_no = Dense(embedding_dim, use_bias=False)
        self.bias_no = tf.Variable(tf.zeros(shape=(embedding_dim,)))
        self.fusion_bias_no = tf.Variable(tf.zeros(shape=(embedding_dim,)))
        self.d6_no = Dense(embedding_dim)
        self.d7_no = Dense(embedding_dim)

        # 不同direction下模型输出的out融合
        self.d8 = Dense(self.hidden_states)
        self.d9 = Dense(self.hidden_states)
        self.d10 = Dense(self.hidden_states)
        self.d11 = Dense(embedding_dim)

        self.conv1 = Conv2D(filters=1, kernel_size=(max_len, embedding_dim), use_bias=False, kernel_regularizer=regularizers.l2(self.l2))
        self.conv2 = Conv2D(filters=1, kernel_size=(max_len, embedding_dim), use_bias=False, kernel_regularizer=regularizers.l2(self.l2))
        self.conv3 = Conv2D(filters=1, kernel_size=(max_len, embedding_dim), use_bias=False, kernel_regularizer=regularizers.l2(self.l2))
        self.d1_last = Dense(1, use_bias=False)
        self.d2_last = Dense(1, use_bias=False)

        # 模型输出之后的全连接层
        self.d4 = Dense(embedding_dim)  # 后接dropout和tf.nn.relu
        self.flat = Flatten()
        self.d5 = Dense(output_dim, use_bias=False, kernel_regularizer=regularizers.l2(self.l2))  # 返回的是没有经过sigmoid的数

        # HAPN的模块
        self.dw1 = Dense(embedding_dim)
        self.dw2 = Dense(embedding_dim)
        self.du1 = Dense(embedding_dim)
        self.du2 = Dense(embedding_dim)

        self.dw1_ex = Dense(embedding_dim)
        self.dw2_ex = Dense(embedding_dim)
        self.du1_ex = Dense(embedding_dim)
        self.du2_ex = Dense(embedding_dim)


    def call(self, inputs):
        f_result, f_score = self.direction_position_embedding(inputs, direction='forward')
        b_result, b_score = self.direction_position_embedding(inputs, direction='backward')
        # no_direction_result, no_score = self.direction_position_embedding(inputs)
        no_direction_result = None
        attn_result = self.output_gate(f_result, b_result, no_direction_result)

        # attn_result = self.sigmoid_fusion_FB(f_result, b_result)
        # output = attn_result

        output = self.FCLayer(attn_result)

        # ========== 与特征工程相匹配，取消最后的全连接层 ============
        # output = attn_result                        # ==========
        # ========== 与特征工程相匹配，取消最后的全连接层 ============

        return output

    def direction_position_embedding(self, inputs, direction=None):
        word_data, position_data, sentence_length = inputs  # 处理以list形式输入的训练样本
        x_train, position_data, sentence_length = inputs   # 加入sentence_length以修正填充式的错误 ----> （改善了mask函数的缺点）
        bs = tf.shape(word_data)[0]
        sll = tf.shape(word_data)[1]

        sl = tf.range(self.max_len)
        word_col, word_row = tf.meshgrid(sl, sl)

        # # ============================== 在softmax时加上高斯先验在===============================
        # d = word_row - word_col
        # d = tf.cast(d, tf.float32) ** 2
        # w = tf.constant(np.random.random(size=d.shape), dtype=tf.float32)
        # b = tf.constant(np.random.random(size=(self.max_len, 1)), dtype=tf.float32) * (-1)
        # bias = tf.matmul(d, w) + b    # (None, max_len, max_len)
        # bias = tf.tile(tf.expand_dims(bias, axis=0), [bs, 1, 1])
        # bias_tile = tf.tile(tf.expand_dims(bias, axis=-1), [1, 1, 1, self.embedding_dim])
        # # ====================================================================================

        if direction == "forward":
            direct_mask = tf.greater(word_row, word_col)
            # direct_mask = 1 - tf.cast(direct_mask, tf.float32)    # 考虑自身
            dense_d1 = self.d1
            dense_d2 = self.d2
            dense_d3 = self.d3
            dense_drop = self.drop
            dense_bias = self.bias
            conv = self.conv1
            dense_d6 = self.d6
            dense_d7 = self.d7
            dense_w1 = self.dw1
            dense_w2 = self.dw2
            dense_u1 = self.du1
            dense_u2 = self.du2

        elif direction == 'backward':
            direct_mask = tf.greater(word_col, word_row)
            # direct_mask = 1 - tf.cast(direct_mask, tf.float32)
            dense_d1 = self.d1_ex
            dense_d2 = self.d2_ex
            dense_d3 = self.d3_ex
            dense_drop = self.drop_ex
            dense_bias = self.bias_ex
            conv = self.conv2
            dense_d6 = self.d6_ex
            dense_d7 = self.d7_ex
            dense_w1 = self.dw1_ex
            dense_w2 = self.dw2_ex
            dense_u1 = self.du1_ex
            dense_u2 = self.du2_ex
        else:
            direct_mask = tf.cast(tf.linalg.tensor_diag(- tf.ones([sll], tf.int32)) + 1, tf.bool)
            dense_d1 = self.d1_no
            dense_d2 = self.d2_no
            dense_d3 = self.d3_no
            dense_drop = self.drop_no
            dense_bias = self.bias_no
            conv = self.conv3
            dense_d6 = self.d6_no
            dense_d7 = self.d7_no

        pad_data = tf.cast(tf.cast(x_train, bool), tf.float32)
        self.pad_data = pad_data

        word_embedding = self.word_embed(word_data)

        position_embedding = self.position_embed(position_data)
        word_embedding = word_embedding + position_embedding  # 加入position信息

        # # ============= 按照Disan的选择，不进行embedding的融合 ========
        # # fixed_embedding = word_embedding + position_embedding   =
        # # fixed_embedding = fx_q(fixed_embedding)                 =
        # # =========================================================
        #
        # # -------------------- 先dropout一下------------------------
        # # word_embedding = self.drop_embedding(word_embedding)
        # # -------------------- 先dropout一下------------------------

        word_embedding1 = dense_d1(word_embedding)
        word_embedding1 = self.selu(word_embedding1)
        word_embedding1 = dense_drop(word_embedding1)
        word_embedding_tile = tf.tile(tf.expand_dims(word_embedding1, 1), [1, sll, 1, 1])
        d2 = dense_d2(word_embedding1)
        d2 = dense_drop(d2)
        d2 = tf.expand_dims(d2, axis=1)
        d3 = dense_d3(word_embedding1)
        d3 = dense_drop(d3)
        d3 = tf.expand_dims(d3, axis=2)
        self_attention_data = d2 + d3 + dense_bias
        logits = fx_q(self_attention_data)

        # # ---------- 使用高斯偏置 -----------
        # # logits = logits + bias_tile
        # # ---------------------------------

        direct_mask_1_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])
        """
        在填充时是存在问题的， 即完全按照pad——data的格式复制填充的话，在列方向上确实填充成功了，
        但是在行方向上，将本不存在的元素也填充为了非0元素
        """
        # 通过下面的操作将pad_data_new 变换成了与sentence_length相同的格式
        # pad_data_tile = tf.tile(tf.expand_dims(pad_data, 1), [1, sll, 1])
        # pad_data_new = tf.tile(tf.expand_dims(pad_data, 2), [1, 1, sll])
        # pad_data_new = tf.cast(pad_data_new, bool)
        # pad_data_tile = tf.cast(pad_data_tile, bool)
        # pad_data_new = tf.logical_and(pad_data_new, pad_data_tile)
        # pad_data_new = tf.cast(pad_data_new, tf.float64).numpy()

        pad_data_tile = sentence_length

        direct_mask_1_tile = tf.cast(direct_mask_1_tile, bool)
        pad_data_tile = tf.cast(pad_data_tile, bool)
        attn_mask = tf.logical_and(direct_mask_1_tile, pad_data_tile)
        self.attn_mask = attn_mask

        # # ============================================ 修改的部分 =======================================
        # #
        # # attn_mask = tf.expand_dims(attn_mask, -1)
        # # +++++    要实现MPSAN的Distance Mask,需要修改attention_mask函数，使得到的logits_mask在logits的基础上
        # # +++++ 不仅只有加0，还可以加  -|i-j| 或者 -log|i-j|
        logits_masked = self.attention_mask(logits, attn_mask)
        attn_score = tf.nn.softmax(logits_masked, 2)
        attn_score = self.position_mask(attn_score, attn_mask)
        attn_result_no_reduce = attn_score * word_embedding_tile
        attn_result = tf.reduce_sum(attn_result_no_reduce, 2)

        # ======================== HAPN =============================
        # word_embedding = self.word_embed(word_data)
        # position_embedding = self.position_embed(position_data)
        # word_embedding = word_embedding + position_embedding
        #
        # word_embedding1 = tf.expand_dims(word_embedding, axis=2)  # (None, 64, 1, 300)
        # word_embedding1 = tf.tile(word_embedding1, [1, 1, self.max_len, self.embedding_dim])  # (None,64,64, 300)
        # word_embedding2 = tf.expand_dims(word_embedding, axis=1)  # (None, 1, 64, 300)  ==> word_specific
        # # word_embedding2 = tf.tile(word_embedding2, [1, ])
        # S_i = dense_w1(word_embedding1) + dense_w2(word_embedding2)  # (None, 64, 64, 300) --> axis=1里面包含了64个不同的S_i
        # # weight_i = tf.nn.tanh(dense_u1(word_embedding1) + dense_u2(S_i))  # (None, 64, 64, 300)
        # weight_i = dense_u1(word_embedding1) + dense_u2(S_i)
        #
        # self_attention_data = d2 + d3 + dense_bias + weight_i
        # logits = fx_q(self_attention_data)
        #
        # logits_masked = self.attention_mask(logits, attn_mask)
        # attn_score = tf.nn.softmax(logits_masked, 2)
        # attn_score = self.position_mask(attn_score, attn_mask)
        # # attn_result_no_reduce = attn_score * S_i
        # attn_result_no_reduce = attn_score * S_i
        # attn_result = tf.reduce_sum(attn_result_no_reduce, 2)
        # #
        # # ============================================ 修改的部分 =======================================

        # attn_result = self.fusion_gate(word_embedding, attn_result, pad_data, direction)
        self.word_embedding = word_embedding

        return attn_result, attn_score

    def FCLayer(self, attn_result):
        attn_result = self.d4(attn_result)
        attn_result = self.drop(attn_result)
        attn_result = tf.nn.relu(attn_result)
        # attn_result = self.selu(attn_result)
        attn_result = self.flat(attn_result)
        out = self.d5(attn_result)
        return out

    def fx_q(self, val, scale=5.):
        return scale * tf.nn.tanh(1. / scale * val)

    def fusion_gate(self, word_embedding, attn_result, pad_data, direction=None):
        if direction == "forward":
            dense_fusion_bias = self.fusion_bias
            dense_d6 = self.d6
            dense_d7 = self.d7
            dense_drop = self.drop
        elif direction == 'backward':
            dense_fusion_bias = self.fusion_bias_ex
            dense_d6 = self.d6_ex
            dense_d7 = self.d7_ex
            dense_drop = self.drop_ex
        else:
            dense_fusion_bias = self.fusion_bias_no
            dense_d6 = self.d6_no
            dense_d7 = self.d7_no
            dense_drop = self.drop_no
        d6 = dense_d6(word_embedding)
        d6 = dense_drop(d6)
        d7 = dense_d7(attn_result)
        d7 = dense_drop(d7)
        fusion_weight = tf.nn.sigmoid(d6 + d7 + dense_fusion_bias)
        out = fusion_weight * word_embedding + (1 - fusion_weight) * attn_result
        output = self.position_mask(out, pad_data)  # 进行填充遮挡
        return output

    def selu(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

    def position_mask(self, embedding, mask):
        mask = tf.expand_dims(mask, -1)
        # mask = mask[tf.newaxis, ...]
        mask = tf.multiply(embedding, tf.cast(mask, tf.float32))
        return mask

    def attention_mask(self, val, attn_mask):
        attn_mask = tf.expand_dims(attn_mask, -1)
        # attn_mask = mask[tf.newaxis, ...]
        attn_mask = (1 - tf.cast(attn_mask, tf.float32)) * (-1e30)
        val = tf.add(val, attn_mask)
        return val

    def sigmoid_fusion_FB(self, f_result, b_result):
        " input: (None, 1, 1, filters), (None, 1, 1, filters)"
        " output: y_predict"
        # filters = f_result.shape[-1]
        # f_result = np.reshape(f_result, [-1, filters])
        # b_result = np.reshape(b_result, [-1, filters])
        f_result = self.flat(f_result)
        b_result = self.flat(b_result)
        f_result = self.d1_last(f_result)
        b_result = self.d2_last(b_result)
        weights = tf.nn.sigmoid(f_result+b_result)
        return weights * f_result + (1-weights) * b_result

    def output_gate(self, out1, out2, out3):
        """
        注释掉的为原来的实现方法，不算正统的把几种attention机制融合的softmax函数，   平均准确率0.921(把pad_mask改正之后的结果，原来的为0.918)
        改写的(此时未注释的)才是正统的方法   （0.928, 此时l2=0.05）  --> (0.9296, 此时l2=0.1)
        """
        # pad_data = self.pad_data
        # out1 = self.drop(self.d8(out1))
        # out2 = self.drop(self.d9(out2))
        # if out3 is None:
        #     fixed_embedding = self.d11(out1 + out2)
        # else:
        #     out3 = self.drop(self.d10(out3))
        #     fixed_embedding = self.d11(out1 + out2 + out3)
        # weights = tf.nn.softmax(fixed_embedding, axis=-1)
        # return weights * fixed_embedding
        pad_data = self.pad_data
        out1 = tf.expand_dims(self.attention_mask(out1, pad_data), axis=1)
        out2 = tf.expand_dims(self.attention_mask(out2, pad_data), axis=1)
        word_embedding = tf.expand_dims(self.attention_mask(self.word_embedding, pad_data), axis=1)
        if out3 is None:
            fixed_embedding = concatenate([out1, out2, word_embedding], axis=1)
        else:
            out3 = tf.expand_dims(self.attention_mask(out3, pad_data), axis=1)
            fixed_embedding = concatenate([out1, out2, out3, word_embedding], axis=1)
        weights = tf.nn.softmax(fixed_embedding, axis=1)
        pad_data = tf.expand_dims(pad_data, axis=1)
        weights = self.position_mask(weights, pad_data)
        return tf.reduce_sum(weights * fixed_embedding, axis=1)

    def output_gate_getweights(self, out1, out2, out3):
        # pad_data = self.pad_data
        # out1 = self.drop(self.d8(out1))
        # out2 = self.drop(self.d9(out2))
        # if out3 is None:
        #     fixed_embedding = self.d11(out1 + out2)
        # else:
        #     out3 = self.drop(self.d10(out3))
        #     fixed_embedding = self.d11(out1 + out2 + out3)
        # # fixed_embedding = self.attention_mask(fixed_embedding, pad_data)
        # weights = tf.nn.softmax(fixed_embedding, axis=-1)
        # # weights = self.position_mask(weights, pad_data)
        # return weights * fixed_embedding, weights
        pad_data = self.pad_data
        out1 = tf.expand_dims(self.attention_mask(out1, pad_data), axis=1)
        out2 = tf.expand_dims(self.attention_mask(out2, pad_data), axis=1)
        word_embedding = tf.expand_dims(self.attention_mask(self.word_embedding, pad_data), axis=1)
        if out3 is None:
            fixed_embedding = concatenate([out1, out2, word_embedding], axis=1)
        else:
            out3 = tf.expand_dims(self.attention_mask(out3, pad_data), axis=1)
            fixed_embedding = concatenate([out1, out2, out3, word_embedding], axis=1)
        weights = tf.nn.softmax(fixed_embedding, axis=1)
        pad_data = tf.expand_dims(pad_data, axis=1)
        weights = self.position_mask(weights, pad_data)
        return tf.reduce_sum(weights * fixed_embedding, axis=1), weights, fixed_embedding

    def mask(self, pad_data):
        mask_row_numble = []
        for i in range(pad_data.shape[0]):  # i个样本     # .shape[0]这一维度为None， 所以不能用
            count = 0
            for j in range(pad_data[i].shape[0]):
                if pad_data[i][j] != 0:
                    count += 1
            mask_row_numble.append(count)
        return mask_row_numble


def main(model_name, dataset):
    import time
    start_time = time.time()

    # max_len = 64
    # max_len = 300
    embedding_dim = 300
    vocab_size = 10000

    # vocab_size = 16789
    # embedding_dim = 200
    max_len = 35

    learning_rate = CustomSchedule(embedding_dim)

    # ========================================== dataset选择 ==================================================
    print("load data:")
    if dataset == " ":
        print("测试模型是否可以运行：")
    elif dataset == "imdb":
        x_train, y_train, x_test, y_test, reverse_word_index = dp.load_imdb(num_words=10000)
        x_train, x_test, max_len = dp.pad_sentence(x_train, x_test, max_len=max_len)
        x_dev, y_dev = x_test, y_test
    elif dataset == "mr":
        data_function = dp_mr.process_data_mr(max_len, embedding_dim)
        x_train, y_train, x_test, y_test, vocab_size, embedding_dim, embedding_matrix, max_len = data_function.split_data()
        x_dev, y_dev = x_test, y_test
    elif dataset == "SST-2":
        # dir1, dir2, dir3 = "SST2_train.txt", "SST2_valid.txt", "SST2_test.txt",
        # x_train, y_train, x_dev, y_dev, x_test, y_test, vocab_size, embedding_matrix = get_all_data(dir1, dir2, dir3,
        #                                                                   max_len,embedding_dim=embedding_dim)  # 50

        train = np.load("train_array.npy")
        x_train = train[:, 1: -1]
        y_train = train[:, 0]

        test = np.load("test_array.npy")
        x_test = test[:, 1: -1]
        y_test = test[:, 0]
        x_dev = x_test
        y_dev = y_test

    elif dataset == "SST-5":
        dir1, dir2, dir3 = "train_five.txt", "valid_five.txt", "test_five.txt",
        x_train, y_train, x_dev, y_dev, x_test, y_test, vocab_size, embedding_matrix = get_all_data(dir1, dir2, dir3,
                                                                                                    max_len,
                                                                                                    embedding_dim=embedding_dim) # 50
        """
        不需要转换为独热编码，tensorflow内部会自动转换的
        """
        # y_train = convert_to_onehot(y_train, 5)
        # y_dev = convert_to_onehot(y_dev, 5)
        # y_test = convert_to_onehot(y_test, 5)

    else:
        raise NotImplementedError

    # ========================================== 模型选择 =====================================================
    if model_name == "lstm":
        model = _LSTM(vocab_size=vocab_size, output_dim=1, embedding_dim=300, max_len=max_len)
    elif model_name == "rnn":
        model = _RNN(vocab_size=vocab_size, output_dim=1, embedding_dim=300, max_len=max_len)
    elif model_name == "textcnn":
        model = TextCNN(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len)
        # model = TextCNN(vocab_size=vocab_size, output_dim=5, embedding_dim=embedding_dim, max_len=max_len)
    elif model_name == "2d_cnn":
        model = _simple2D_CNN(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len)
    elif model_name == "1d_cnn":
        model = _1D_CNN(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len)
    elif model_name == "target-attention":
        model = only_attention(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len)
    elif model_name == "self-attention":
        model = self_attention(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len,
                               attn_mask=True)
    elif model_name == "self-attention-mask":
        model = self_attention_mask(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len,
                                    attn_mask=False)
    elif model_name == 'disan-attention':
        x_train, x_dev, x_test = change_train_data(x_train, x_dev, x_test, max_len)  # 此模型需要修改输入数据的形式
        model = position_attention(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                      loss=losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # model = position_attention(vocab_size=vocab_size, output_dim=5, embedding_dim=embedding_dim, max_len=max_len)
        # model.compile(optimizer=optimizers.Adam(learning_rate),
        #               loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        #               metrics=['accuracy'])
    else:
        raise NotImplementedError

    if dataset == " ":
        print("模型可以运行， 但未导入数据！")
        exit(0)

    # plot_model(model, to_file=model_name + ".png", show_shapes=True)

    # lrm = LearningRateScheduler(lr_)
    # early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    # history = model.fit(x_train, y_train, epochs=10, batch_size=256,
    #                     validation_data=(x_dev, y_dev), callbacks=[early_stop, lrm], verbose=2)

    history = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=(x_dev, y_dev), verbose=1)  # (40, 256)
    # model.summary()
    model.evaluate(x_test, y_test, verbose=2)
    # _plot = pl(history)

    acc =history.history["val_acc"]
    acc = np.array(acc)
    print("最大的准确率为 ： " + str(acc.max()))

    # filepath = 'saved_model'
    # model.save(filepath)

    end_time = time.time()
    print(end_time - start_time)
    print()


if __name__ == '__main__':
    # main("target-attention", "imdb")

    # main("textcnn", "imdb")
    # main("rnn", "imdb")
    # main("lstm", "imdb")
    # main("self-attention", "imdb")

    main("target-attention", "imdb")
