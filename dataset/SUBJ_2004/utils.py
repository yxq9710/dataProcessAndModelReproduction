import tensorflow as tf
from codes.Dcnn_add_Dcnn.function_test import position_attention, CustomSchedule
from codes.Dcnn_add_Dcnn.testModel import testModel
from tensorflow.keras import losses, optimizers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


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
    indexes = indexes.tolist()
    for i in range(len(indexes)):
        index = indexes[i]
        word = index_word[index]
        indexes[i] = word
    words = indexes
    return words


def get_id2word(vocab):
    id2word = {}
    id2word[0] = ''
    for word, id in vocab.items():
        id2word[id] = word
    return id2word


def mask(pad_data):
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
    # inputs = change_evaluate_data(sentence, max_len)
    inputs, _, _ = change_train_data(sentence, sentence, sentence, max_len)

    # f_result, f_score = model.direction_position_embedding(inputs, direction='forward')
    # b_result, b_score = model.direction_position_embedding(inputs, direction='backward')
    # no_direction_result = None
    #
    # attn_result, weights, fixed_embedding = model.output_gate_getweights(f_result, b_result, no_direction_result)

    attn_result, weights_i = model.attention(inputs)
    output = model.FCLayer(attn_result)
    output = tf.nn.sigmoid(output)
    print(np.squeeze(output))

    weights = weights_i  # (1, 64, 64, 300)
    weights = np.squeeze(weights)
    weights = np.sum(weights, axis=-1)
    weights_t = np.transpose(weights)
    weights_new = np.matmul(weights, weights_t) / (np.linalg.norm(weights_t) * np.linalg.norm(weights))
    # weights = tf.nn.softmax(weights, axis=1)  # (1, 64, 64, 300)
    # for()
    # weights = np.sum(weights, axis=-1)

    # weights = np.reshape(weights, [-1, 1])

    attention_plot = np.zeros([max_len, max_len])
    for i in range(max_len):
        attention_plot[i] = weights_new[i]

    id2word = get_id2word(vocab)
    sentence = to_word(sentence, id2word)
    plot_attention(attention_plot, sentence)

    return output


def plot_attention(attention, sentence):
    count = 0
    for a in sentence:
        if a != '':
            count += 1
    del sentence[count:]  # 根据索引删除
    attention = attention[:count, :count]

    predicted_sentence = sentence
    fig = plt.figure(figsize=(count, count))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 60}  # 设置字体大小

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)  # 设置x轴的文字和标签，并且翻转90度
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # 将主刻度标签设置为1的倍数
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def get_error_sentence(dir, encode='ISO-8859-1'):
    sentences = []
    with open(dir, 'r', encoding=encode)as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            line[0] = line[0].strip().split()
            sentences.append(line[0])
        f.close()
    return sentences


def pad_list_word(sentences):
    digit_sentences = []
    for sentence in sentences:
        digit_sentence = to_index(sentence, vocab)
        digit_sentences.append(digit_sentence)
    digit_sentences = tf.keras.preprocessing.sequence.pad_sequences(digit_sentences, max_len, value=0, padding='post')
    return digit_sentences


if __name__ == '__main__':
    # import sys
    # print(sys.path)
    vocab_size = 23906
    embedding_dim = 300
    max_len = 64
    # model = position_attention(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len)
    model = testModel(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len)

    vocab = get_vocab(dir1='plot.tok.gt9.5000', dir2='quote.tok.gt9.5000', dir3='vocab.txt')

    i = 0
    checkpoint_filepath = 'results2/checkpoint/checkpoint'
    filepath = checkpoint_filepath + '_' + str(i)
    model.load_weights(filepath)
    # model.compile(optimizer=optimizers.Adam(),  # 静态学习率没有动态学习率更适应模型
    #               loss=losses.BinaryCrossentropy(from_logits=True),
    #               metrics=['accuracy'])

    sentences = get_error_sentence(dir='E:\sentiment_classification\dataset\SUBJ_2004\\results2\error_sample_' + str(i) + '.txt')
    sentences = pad_list_word(sentences)
    predicts = np.zeros(shape=[sentences.shape[0], 1])
    for i, sentence in enumerate(sentences):
        # s = "terry , a spoilt brat is just too lazy a student ."
        s = "amitabh can't believe the board of directors and his mind is filled with revenge and what better revenge than robbing the bank himself , ironic as it may sound ."
        line = s.strip().split('\t')
        s = line[0].strip().split()
        digit_sentence = to_index(s, vocab)
        item = [0]
        for i in range(64-len(digit_sentence)):
            digit_sentence += item
        digit_sentence = np.array(digit_sentence)
        probability = Evaluate_error(model, digit_sentence, vocab, max_len)
        probability = Evaluate_error(model, sentence, vocab, max_len)
        predicts[i] = np.array(probability)

