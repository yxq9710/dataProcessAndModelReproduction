import tensorflow as tf
from codes.Dcnn_add_Dcnn.function_test import position_attention
from tensorflow.keras import losses, optimizers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorflow.keras.utils import to_categorical


def get_vocab(dir1, dir2, dir3, encode='ISO-8859-1'):
    vocab = {}
    vocabs = {}
    for dir in [dir1, dir2]:
        with open(dir, 'r', encoding=encode) as f:
            lines = f.readlines()
            for line in lines:
                line = line.lower().strip().split()
                for line_word in line:
                    if line_word not in vocabs.keys():
                        # if line_word != vocab_word:
                        vocabs[line_word] = 1
                    else:
                        vocabs[line_word] += 1
            f.close()

    vocabs = sorted(vocabs.items(), key=lambda item: item[1], reverse=True)
    id = 1
    for line_word, count in vocabs:
        vocab[line_word] = id
        id += 1

    with open(dir3, 'w', encoding=encode) as file:
        for line_word in vocab.keys():
            file.writelines(line_word + '\t' + str(vocab[line_word]) + '\n')
        file.close()
    return vocab


def to_index(words, word_to_index):
    """
    将一个由word组成的list转化为glove里对应的index
    """
    for i in range(len(words)):
        word = words[i]
        if word in word_to_index.keys():
            index = word_to_index[word]
        else:
            index = 0
        words[i] = index
    indexes = words
    return indexes


def to_word(indexes, index_word):
    """
    translate a indexes'list to words by index_word

    input: indexes: 索引列表
           index_word: index_word形式的词表  来自于数据集的vocab
    output: words: 索引对应的words
    """
    if isinstance(indexes, list):
        return indexes
    else:
        indexes = indexes.tolist()
        for i in range(len(indexes)):
            index = indexes[i]
            word = index_word[index]
            indexes[i] = word
        words = indexes
        return words


def get_id2word(vocab):
    """
    将vocab转化为id2word
    """
    id2word = {}
    id2word[0] = ''
    for word, id in vocab.items():
        id2word[id] = word
    return id2word


def mask(pad_data):
    """
    记录每个sentence的非填充的word的个数
    input:   pad_data（np.array）: 待输入模型的句子(x_train的data) , shape: (max_len, ) 或者（batch_size, max_len）
    output： mask_numble (np.array) : 记录每个sentence的mask个数的数组 ， shape：（batch_size, 1）
    """

    if pad_data.ndim != 2:
        pad_data = np.reshape(pad_data, [1, -1])
    mask_row_numble = []
    for i in range(pad_data.shape[0]):  # i个样本
        count = 0
        for j in range(pad_data[i].shape[0]):
            if pad_data[i][j] != 0:
                count += 1
        mask_row_numble.append(count)
    return np.reshape(np.array(mask_row_numble), [-1, 1])


def process_mask(mask_length, max_len):
    """
    将mask后的数组处理为在模型中可以直接使用的形式
    input: mask_numble (np.array), shape: （batch_size, 1）
    output: pad_data_tile(np.array), shape: (batch_size, max_len, max_len)
    """
    pad_data_tile = np.zeros([mask_length.shape[0], max_len, max_len])
    for i in range(len(mask_length)):
        m = np.squeeze(mask_length[i])
        a = np.ones(shape=(m, m))
        pad_data_tile[i][:m, :m] = a
    return pad_data_tile


def change_train_data(x_train, x_dev, x_test, max_len):
    """
    这种掩码才是正确的
    input: np.array: x_train
    result: a list:  [x_train, position_encoding, pad_mask]
    """
    if x_train.ndim == 1 and x_dev.ndim == 1 and x_test.ndim == 1:
        x_train = np.reshape(x_train, [1, -1])
        x_dev = np.reshape(x_dev, [1, -1])
        x_test = np.reshape(x_test, [1, -1])

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


def change_evaluate_data(data, max_len):
    """
    这种封装的数据没有把行掩码完全考虑进去，有错误
    """
    # data = np.reshape(data, [1, -1])
    a = tf.range(max_len)
    b = tf.constant(a, shape=(1, max_len))
    c_train = tf.tile(b, [data.shape[0], 1]).numpy()
    data = [data, c_train]
    return data


def Evaluate_error(model, sentence, vocab, max_len):
    """
    计算错误样本的predict值，并且会绘制其注意力权值图
    input: model    : 训练后的模型
           sentence : 错误样本
           vocab    : 词表
           max_len  : 填充后的句子长度
    output: 样本类别的predict值
    """

    # inputs = change_evaluate_data(sentence, max_len)
    inputs, _, _ = change_train_data(sentence, sentence, sentence, max_len)

    f_result, f_score = model.direction_position_embedding(inputs, direction='forward')
    b_result, b_score = model.direction_position_embedding(inputs, direction='backward')
    # no_direction_result, no_score = model.direction_position_embedding(inputs)
    no_direction_result = None
    # attn_result = concatenate([f_result, b_result, no_direction_result], axis=1)
    attn_result, weights = model.output_gate_getweights(f_result, b_result, no_direction_result)
    output = model.FCLayer(attn_result)
    output = tf.nn.sigmoid(output)
    print(np.squeeze(output))

    # a = model.predict(inputs)
    # b = tf.nn.sigmoid(a)
    # print()

    # for weights in [f_score, b_score]:
    #     # weights = f_score      # (1, 64, 64, 300)
    #     weights = np.squeeze(weights)
    #     weights = np.linalg.norm(weights, axis=-1)
    #     # weights = tf.nn.softmax(weights, axis=1)  # (64, 64)
    #     weights = tf.nn.sigmoid(weights)
    #     weights = np.exp(weights)
    #
    #     # weights = np.reshape(weights, [-1, 1])
    #
    #     attention_plot = np.zeros_like(weights)
    #     for i in range(weights.shape[0]):
    #         attention_plot[i] = weights[i]
    #
    #     id2word = get_id2word(vocab)
    #     sentence = to_word(sentence, id2word)
    #     plot_attention(attention_plot, sentence)

    return output


def plot_attention(attention, sentence):
    """
    绘制样本的注意力权值图
    input: attention: 注意力权值矩阵
           sentence: 样本
    output: no
    """

    count = 0
    for a in sentence:
        if a != '':
            count += 1
    del sentence[count:]
    attention = attention[:count, :count]

    predicted_sentence = sentence
    fig = plt.figure(figsize=(count, count))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    # ax.matshow(attention, cmap='plasma')
    # ax.matshow(attention, cmap='inferno')
    # ax.matshow(attention, cmap='magma')
    # ax.matshow(attention, cmap='cividis')

    fontdict = {'fontsize': 60}    # 设置字体大小

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)  # 设置x轴的文字和标签，并且翻转90度
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # 将主刻度标签设置为1的倍数
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    print()


def get_error_sentence(dir, encode='ISO-8859-1'):
    """
    读取错误样本
    input: dir: 样本所在文本的路径
           encode:编码方式
    output: 有错误样本组成的list, 每个元素为一个样本
    """
    sentences = []
    with open(dir, 'r', encoding=encode)as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            line[0] = line[0].strip().split()
            sentences.append(line[0])
        f.close()
    return sentences


def pad_list_word(sentences, vocab, max_len):
    """
    将样本由word转化为index，并填充
    input: sentences: a list, 样本列表
           vocab: 词表
           max_len: 。。。
    output: digit_sentences: a list, 填充并转化后的样本列表
    """
    digit_sentences = []
    for sentence in sentences:
        digit_sentence = to_index(sentence, vocab)
        digit_sentences.append(digit_sentence)
    digit_sentences = tf.keras.preprocessing.sequence.pad_sequences(digit_sentences, max_len, value=0, padding='post')
    return digit_sentences


def get_relative_position_mask(distance, max_len):
    """
    创建一个distance范围内有效的掩码
    input: distance: 与中心word相关的距离
           max_len: ...
    output: 只有distance内的word有效的mask矩阵
    """
    a = np.eye(max_len)
    m = range(distance+1)
    for i in range(1, len(m)):
        n = np.eye(max_len, k=m[i])
        a += n
    row, col = np.diag_indices_from(a)   # 找到对角线的索引
    a[row, col] = 0
    return a


def modify_false_sample(i, probability):
    line_i = []
    with open("errors_4/error_sample_" + str(i) + ".txt", 'r', encoding="ISO-8859-1")as f:
        lines = f.readlines()
        for id, line in enumerate(lines):
            line = line.strip().split('\t')

            # ============== 把p与t之差大于0.9的样本挑出来 =================
            # if np.abs(float(line[1]) - float(line[3])) >= 0.9:
            #     line1 = line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\n'
            #     line_i.append(line1)
            # ==========================================================

            line1 = line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + str(np.squeeze(probability[id])) + '\n'
            line_i.append(line1)
        f.close()
    with open("errors_4/error_sample_" + str(i) + "_extract.txt", 'w', encoding="ISO-8859-1")as f:
        for line1 in line_i:
            f.writelines(line1)
        f.close()


def convert_to_onehot(labels, num_classes):
    return to_categorical(labels, num_classes)


# y = np.array([1, 2, 4])
# print(y)
# c = convert_to_onehot(y, 5)
# print(c)