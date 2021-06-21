"""
    这种构建此表的方式有很多缺陷，因为分词方式有问题，导致 ’，‘ ’，- 等符号产生错误分词，都会造成噪音

    在class里面写泛化后的计算公式，不能写具体的小的加减乘除之类的，不能会报 NoneType 相关的错误 ----解决思路是在外面写好，
        然后都转换为np.array的格式，封装在x_train里面输入模型
"""

import tensorflow as tf
import numpy as np

from codes.Dcnn_add_Dcnn.utils import plot_attention
from dataset.SUBJ_2004.utils import get_id2word
from dataset.pre_trained.embedding_matrix_glove import to_word

dir1 = 'processed_plot.tok.gt9.5000'
dir2 = 'plot_1.txt'
vocab_from_nltk = False


def generate_txt(dir1='plot.tok.gt9.5000', dir2='plot_1.txt', label=1):
    word_label = []
    with open(dir1, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            word_label.append(line + '\t' + str(label) + '\n')
        f.close()

    with open(dir2, 'w', encoding='ISO-8859-1') as file:
        file.writelines(word_label)
        file.close()


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


def from_txt_get_vocab(dir1='new_vocab.txt', encode='ISO-8859-1'):
    vocab = {}  # word2id
    with open(dir1, 'r', encoding=encode)as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split()
            vocab[line[0]] = int(line[1])
        f.close()
    return vocab


if vocab_from_nltk:
    generate_txt(dir1='processed_plot.tok.gt9.5000', dir2='plot_1.txt', label=1)
    generate_txt(dir1='processed_quote.tok.gt9.5000', dir2='quote_0.txt', label=0)
    vocab = from_txt_get_vocab(dir1='vocab_nltk_re.txt')
else:
    generate_txt()
    generate_txt(dir1='quote.tok.gt9.5000', dir2='quote_0.txt', label=0)
    vocab = get_vocab(dir1='plot.tok.gt9.5000', dir2='quote.tok.gt9.5000', dir3='vocab.txt')


id2word = {}
id2word[0] = ''
for word, id in vocab.items():
    id2word[id] = word


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


def process_txt(dir1, encode=None):
    words = []
    labels = []
    if encode is None:
        encode = 'utf-8'
    with open(dir1, 'r', encoding=encode) as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().strip('\n').split('\t')
            line_word = line[0].lower().strip().split()
            words.append(line_word)
            label = int(line[1])
            labels.append(label)
        file.close()
        return words, labels


def get_data(dir1, word_to_index, encode=None):
    train_words, train_labels = process_txt(dir1, encode)
    for i, words in enumerate(train_words):
        data = to_index(words, word_to_index)
        words = data
        words[0:0] = (train_labels[i], )
    return train_words


def get_shuffled_data(dataset_i):
    labels = []
    datasets = []
    for subList in dataset_i:
        labels.append(subList[0])
        datasets.append(subList[1:])
    return datasets, labels


sentences_1 = get_data(dir1='plot_1.txt', word_to_index=vocab, encode='ISO-8859-1')
sentences_0 = get_data(dir1='quote_0.txt', word_to_index=vocab, encode='ISO-8859-1')

"""
 接下来：设计 十折交叉验证 划分数据集
    列表拼接： 用 +  ： list(a) + list(b)
"""
import math, random

lens = len(sentences_1)
k = 10
batch_num = math.floor(lens / k)
print(batch_num)

dataset = {}
labels = {}
for i in range(k):
    dataset[i] = sentences_1[i * batch_num: (i + 1) * batch_num] + sentences_0[i * batch_num: (i + 1) * batch_num]
    random.shuffle(dataset[i])
    dataset[i], labels[i] = get_shuffled_data(dataset[i])

from codes.Dcnn_add_Dcnn.function_test import position_attention, CustomSchedule, self_attention
from tensorflow.keras import losses, optimizers

vocab_size = len(vocab)
embedding_dim = 300
max_len = 64
batchsz = 32
epoch = 30
warmup_steps = 12
learning_rate = CustomSchedule(embedding_dim, warmup_steps=4000)

dataset_train = {}
dataset_test = {}
for _iter in range(k):
    x_train = []
    y_train = []
    for i in range(k):
        if i == _iter:
            x_test = dataset[i]
            y_test = labels[i]
        else:
            x_train += dataset[i]
            y_train += labels[i]
    dataset_train[_iter] = (x_train, y_train)
    dataset_test[_iter] = (x_test, y_test)


accuracy = {}

for i in range(k):
    (x_train, y_train) = dataset_train[i]
    (x_test, y_test) = dataset_test[i]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, value=0, padding='post', maxlen=max_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, value=0, padding='post', maxlen=max_len)

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


    x_train, x_test, x_test = change_train_data(x_train, x_test, x_test, max_len)

    weight_path = 'saved_weights/weights_' + str(i)

    checkpoint_filepath = 'checkpoint/checkpoint'
    filepath = checkpoint_filepath + '_' + str(i)

    model = position_attention(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len)

    # # ====================================== 观察注意力权重 ======================================================
    # x_train = x_train[: 256]
    # y_train = y_train[: 256]
    # x_test = x_test[: 256]
    # y_test = y_test[:256]
    # vocab = get_vocab(dir1='plot.tok.gt9.5000', dir2='quote.tok.gt9.5000', dir3='vocab.txt')
    # model, model1 = self_attention(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len)
    # history = model.fit(x_train, y_train, epochs=3, verbose=1, validation_data=(x_test, y_test), batch_size=128)
    #
    # for train_data in x_train:
    #     train_data = np.reshape(train_data, [-1, max_len])
    #     weights = model1.predict(train_data)
    #     weights = np.squeeze(weights)
    #     weights = np.sum(weights, axis=-1)
    #     # weights = np.linalg.norm(weights, axis=-1)
    #     attention_plot = np.zeros([max_len, max_len])
    #     for i in range(max_len):
    #         attention_plot[i] = weights[i]
    #
    #     id2word = get_id2word(vocab)
    #     train_data = np.squeeze(train_data)
    #     train_data = train_data.tolist()
    #     sentence = to_word(train_data, id2word)
    #     plot_attention(attention_plot, sentence)
    # # ====================================== 观察注意力权重 ======================================================

    # model.load_weights(filepath)


    model.compile(optimizer=optimizers.Adadelta(learning_rate=0.5),  # 静态学习率没有动态学习率更适应模型
                    loss=losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_test, y_test), batch_size=8)

    loss, accuracy[i] = model.evaluate(x_test, y_test, verbose=2)

    save_error_sample = False
    if save_error_sample:
        y_predict = model.predict(x_test)
        y_predict = tf.nn.sigmoid(y_predict).numpy()
        y_predict = np.squeeze(y_predict)
        y_predict = tf.stack([1 if value > 0.5 else 0 for value in y_predict]).numpy()

        samples = (x_test, y_test, y_predict)
        false_sample = []
        counter = 0
        for _iter in range(y_test.shape[0]):
            p = y_predict[_iter]
            t = y_test[_iter]
            if p != t:
                counter += 1
                false_sample.append((x_test[0][_iter], t, p))

        with open('error_sample_' + str(i) + '.txt', 'w', encoding='ISO-8859-1') as f:
            for sent_tuple in false_sample:
                sent_id, t_label, p_label = sent_tuple
                # sent_id = sent_id.tolist()
                sent_id = to_word(sent_id, id2word)
                sent_id = ' '.join(word for word in sent_id)
                f.writelines(sent_id + '\t' + str(t_label) + '\t' + str(p_label) + '\n')
            f.close()

    # model.save_weights(weight_path)


len_accuracy = len(accuracy)
sum = 0
for i in range(len_accuracy):
    sum += accuracy[i] / len_accuracy

print(sum)


