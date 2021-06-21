from tensorflow import keras
import numpy as np
from dataset.pre_trained.embedding_matrix_glove import pretrained, to_index


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


def get_words(dir1, dir2, dir3):
    train_words, train_labels = process_txt(dir1)
    dev_words, dev_labels = process_txt(dir2)
    test_words, test_labels = process_txt(dir3)
    return train_words, train_labels, dev_words, dev_labels, test_words, test_labels


def get_data(dir1, word_to_index):
    train_words, train_labels = process_txt(dir1)
    for words in train_words:
        data = to_index(words, word_to_index)
        words = data
    return np.array(train_words), np.array(train_labels)


def pad_sentence(x_train, x_dev, x_test, max_len=64):
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, value=0, padding='post', maxlen=max_len)
    x_dev = keras.preprocessing.sequence.pad_sequences(x_dev, value=0, padding='post', maxlen=max_len)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, value=0, padding='post', maxlen=max_len)
    return x_train, x_dev, x_test


def get_all_data(dir1, dir2, dir3, max_len, embedding_dim):
    glove = pretrained(embedding_dim)
    x_train, labels_train = get_data(dir1, glove.word_to_index)
    x_dev, labels_dev = get_data(dir2, glove.word_to_index)
    x_test, labels_test = get_data(dir3, glove.word_to_index)
    x_train, x_dev, x_test = pad_sentence(x_train, x_dev, x_test, max_len)
    return x_train, labels_train, x_dev, labels_dev, x_test, labels_test, glove.vocab_size, glove.embedding_matrix


def main():
    dir1 = "SST2_train.txt"
    words, labels = process_txt(dir1)
    get_data(50, words)
