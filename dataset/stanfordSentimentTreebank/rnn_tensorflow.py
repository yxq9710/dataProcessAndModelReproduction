import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import numpy as np

os.environ['KMP_WARNINGS'] = 'FALSE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def category_binary(dir_r, dir_w):
    """
        将5分类标签变为2分类
    注意：此时将分类为neutral的结果剔除
    """

    with open(dir_r, 'r', encoding='utf-8') as file_r:
        lines = file_r.readlines()
        with open(dir_w, 'w', encoding='utf_8') as file_w:
            for line in lines:
                line = line.strip().split('\t')
                if float(line[1]) >= 0.6:
                    line[1] = 1
                elif float(line[1]) < 0.4:
                    line[1] = 0
                else:
                    continue
                file_w.write(line[0] + '\t' + str(line[1]) + '\n')
            file_w.close()
        file_r.close()


def category_five(dir_r, dir_w):
    """
        5分类标签 : 此标签顺序与.pickle文件读取的标签类别相同

    """

    with open(dir_r, 'r', encoding='utf-8') as file_r:
        lines = file_r.readlines()
        with open(dir_w, 'w', encoding='utf_8') as file_w:
            for line in lines:
                line = line.strip().split('\t')
                if float(line[1]) > 0.8:
                    line[1] = 4
                elif float(line[1]) > 0.6:
                    line[1] = 3
                elif float(line[1]) > 0.4:
                    line[1] = 2
                elif float(line[1]) > 0.2:
                    line[1] = 1
                else:
                    line[1] = 0
                file_w.write(line[0] + '\t' + str(line[1]) + '\n')
            file_w.close()
        file_r.close()


# category_binary('train_final.txt', 'train_binary.txt')
# category_binary('valid_final.txt', 'valid_binary.txt')
# category_binary('test_final.txt', 'test_binary.txt')
category_five('train_final.txt', 'train_five.txt')
category_five('valid_final.txt', 'valid_five.txt')
category_five('test_final.txt', 'test_five.txt')
print()


def load_word_index(embedding_dim):
    word_to_vector = {}
    with open('E:\myWorks\dataset\pre_trained\glove.6B\glove.6B.' + str(embedding_dim) + 'd.txt', 'r', encoding='utf-8') as glove:
        lines = glove.readlines()
        for line in lines:
            line = line.lower().strip().split()
            word_to_vector[line[0]] = np.array(line[1:], dtype=np.float64)
        glove.close()
    return word_to_vector


def word_data_to_vector(dir, word_length, embedding_dim, word_to_vector):
    x_train = []
    y_train = []
    vector = np.zeros([word_length, embedding_dim])   # 每次截取前10个word， 每个word为50维的数据 [word_length, embedding-dim]
    with open(dir, 'r', encoding='utf-8') as word_data:
        # 'train_binary.txt'
        lines = word_data.readlines()
        for line in lines:
            line = line.strip().split('\t')
            line_word = line[0].lower().strip().split()
            for i, word in enumerate(line_word):
                if i < word_length:   # 截取
                    if word == '' or word not in word_to_vector.keys():  # 填充
                        vector[i] = np.zeros([1, embedding_dim])
                    else:
                        vector[i] = word_to_vector[word]
            x_train.append(vector)
            y_train.append(int(line[1]))
        word_data.close()
    return np.array(x_train), np.array(y_train)


def get_data(train_dataset, dev_dataset, test_dataset, word_length, embedding_dim):
    word_to_vector = load_word_index(embedding_dim)
    x_train, y_train = word_data_to_vector(train_dataset, word_length, embedding_dim, word_to_vector)
    x_dev, y_dev = word_data_to_vector(dev_dataset, word_length, embedding_dim, word_to_vector)
    x_test, y_test = word_data_to_vector(test_dataset, word_length, embedding_dim, word_to_vector)
    return x_train, y_train, x_dev, y_dev, x_test, y_test


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        self.rnn = Sequential([
            layers.SimpleRNN(units, dropout=0.3, return_sequences=True),
            layers.SimpleRNN(units, dropout=0.3)
        ])
        self.out_layer = Sequential([
            layers.Dense(32),   # 影响第二层的参数数量
            layers.Dropout(rate=0.3),
            layers.ReLU(),
            layers.Dense(1)
        ])

    def call(self, inputs, training=None):
        x = inputs
        x = self.rnn(x)
        x = self.out_layer(x, training)
        prob = tf.sigmoid(x)
        return prob


def main():

    word_length = 20
    embedding_dim = 50
    word_to_vector = load_word_index(embedding_dim)
    x_train, y_train = word_data_to_vector('train_binary.txt', word_length, embedding_dim, word_to_vector)
    x_dev, y_dev = word_data_to_vector('valid_binary.txt', word_length, embedding_dim, word_to_vector)
    x_test, y_test = word_data_to_vector('test_binary.txt', word_length, embedding_dim, word_to_vector)

    # x_train, y_train, x_dev, y_dev, x_test, y_test = \
    #     get_data('train_binary.txt', 'valid_binary.txt', 'test_binary.txt', word_length, embedding_dim)

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train = train.shuffle(1000).batch(128, drop_remainder=True)
    dev = tf.data.Dataset.from_tensor_slices((x_dev, y_dev))
    dev = dev.shuffle(1000).batch(128, drop_remainder=True)
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test = test.batch(128, drop_remainder=True)

    units = 32   # 影响两层的参数数量
    epoches = 20
    model = MyRNN(units)

    model.compile(optimizer=optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train, epochs=epoches, validation_data=dev, verbose=2)
    model.evaluate(test, verbose=2)
    model.summary()


if __name__ == '__main__':
    main()

