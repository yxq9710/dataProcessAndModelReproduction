import tensorflow as tf
from tensorflow import keras
import numpy as np


def decode_review(text):
    return ''.join([])


(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
word_index = keras.datasets.imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
"""前四个是特殊位"""
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for key, value in word_index.items()])

vocabs = sorted(word_index.items(), key=lambda item: item[1])
with open('vocab.txt', 'w', encoding='utf-8') as f:
    for key, value in vocabs:
        f.writelines(key + '\t' + str(value) + '\n')
    f.close()

print()


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# print(x_train.shape, len(x_train[0]), y_train.shape)
# print(decode_review(x_train[0]))
# print(decode_review(x_test[0]))


# def dataprocess(dir):
#     with open(dir, 'w', encoding='utf-8') as trainfile:
#         for i in range(len(x_train)):
#             a = decode_review(x_train[i])
#             trainfile.write(a + '\n')
#         trainfile.close()
#
#
# dataprocess('train.txt')
# dataprocess('test.txt')

# with open('train.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     print(len(lines))


