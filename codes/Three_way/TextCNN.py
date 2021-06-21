# %%
import sys
sys.path.append("/codes/bilstm_crf_CNN")
print(sys.path)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Embedding, concatenate, Flatten, MaxPool1D
from tensorflow.keras import Input
import logging
import os
import matplotlib.pyplot as plt
from tensorflow.keras.constraints import max_norm
# from plt_loss import plot_loss_and_accuracy
# from load_type_data import load_train_data, load_test_data
# from bilstm_crf import main
# from dataset.pre_trained import embedding_matrix_glove as em


os.environ['FORCE_GPU_ALLOW_GROWTH'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

max_len = 64
vocab_size = 10000
output_dim = 1
embedding_dim = 100
embedding_matrix = None
batchsz = 64


# %% load data
imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print(len(x_train[0]))
# print(len(x_test[0]))
# print(type(x_test))
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
# print(word_index)
word_index['<PAD>'] = 0
word_index['<START'] = 1
word_index['UNK'] = 2
word_index['UNUSED'] = 3
print(x_train[0])

# ======================== use embedding_matrix ======================
# index_word = {v: k for k, v in word_index.items()}
# em_m = em.pretrained(embedding_dim)
# embedding_matrix = em_m.embedding_matrix
# for i in range(len(x_train)):
#     x_train[i] = em.to_word(x_train[i], index_word)
#     x_train[i] = em.to_index(x_train[i], em_m.word_to_index)
# for i in range(len(x_test)):
#     x_test[i] = em.to_word(x_test[i], index_word)
#     x_test[i] = em.to_index(x_test[i], em_m.word_to_index)
# ====================================================================

# %% data process
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, value=0, padding='post',
                                                        maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, value=0, padding='post',
                                                       maxlen=max_len)
print(len(x_train[0]))
print(x_train[0])
print(y_train[0])
print(x_train[0].shape)


# %%
def TextCNN(vocab_size, output_dim, embedding_dim, embedding_matrix=None):
    x_input = Input(shape=(max_len, ))
    if embedding_matrix is None:
        x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(x_input)
    else:
        x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len,
                      weights=[embedding_matrix], trainable=False)(x_input)
    x = x[..., tf.newaxis]
    filters = [100, 100, 100]
    output_pool = []
    kernel_sizes = [3, 4, 5]
    for i, kernel_size in enumerate(kernel_sizes):
        conv = Conv2D(filters=filters[i], kernel_size=(kernel_size, embedding_dim),
                      padding='valid', kernel_constraint=max_norm(3, [0, 1, 2]))(x)
        pool = MaxPool2D(pool_size=(max_len-kernel_size+1, 1))(conv)
        # pool = tf.keras.layers.GlobalAveragePooling2D()(conv)  # 1_max pooling
        output_pool.append(pool)
        # logging.info("kernel_size: {}, conv.shape: {}, pool.shape: {}".format(kernel_size, conv.shape, pool.shape))
        print("kernel_size: {}, conv.shape: {}, pool.shape: {}".format(kernel_size, conv.shape, pool.shape))
    output_pool = concatenate([p for p in output_pool])
    # logging.info("output_pool.shape: {}".format(output_pool.shape))
    print("output_pool.shape: {}".format(output_pool.shape))

    x = Dropout(rate=0.5)(output_pool)
    x = Flatten()(x)
    y = Dense(output_dim, activation='sigmoid')(x)
    model = tf.keras.Model([x_input], y)
    model.summary()
    return model


model = TextCNN(vocab_size, output_dim, embedding_dim, embedding_matrix)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# model_0 = TextCNN(vocab_size, output_dim, embedding_dim, embedding_matrix)
# model_1 = TextCNN(vocab_size, output_dim, embedding_dim, embedding_matrix)
# model_2 = TextCNN(vocab_size, output_dim, embedding_dim, embedding_matrix)


# %%
# model_0.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model_1.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model_2.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# NER_model_epoch = 20
# # train_x_0, train_y_0, train_x_1, train_y_1, train_x_more, train_y_more = load_train_data(NER_model_epoch)
# (train_x_0, train_x_1, train_x_more, train_y_0, train_y_1, train_y_more), (test_x_0, test_x_1, test_x_more, test_y_0, test_y_1, test_y_more) = main()
# train_x_data = [train_x_0, train_x_1, train_x_more]
# train_y_data = [train_y_0, train_y_1, train_y_more]
# # test_x_0, test_y_0, test_x_1, test_y_1, test_x_more, test_y_more = load_test_data(NER_model_epoch)
# test_x_data = [test_x_0, test_x_1, test_x_more]
# test_y_data = [test_y_0, test_y_1, test_y_more]
# len_0 = test_x_0.shape[0]
# len_1 = test_x_1.shape[0]
# len_more = test_x_more.shape[0]
# len_sample = [len_0, len_1, len_more]
# historys = []
# losses = []
# accuracys = []
# acc = 0
# models = [model_0, model_1, model_2]
# Epochs = [5, 5, 5]   # 79%
# for i in range(3):
#     history = models[i].fit(train_x_data[i], train_y_data[i], epochs=Epochs[i],
#                             batch_size=batchsz, validation_data=(test_x_data[i], test_y_data[i]), verbose=2)
#     loss, accuracy = models[i].evaluate(test_x_data[i], test_y_data[i])
#     acc += accuracy * len_sample[i]
#     plot_loss_and_accuracy(history)
#     historys.append(history)
#     losses.append(loss)
#     accuracys.append(accuracy)
# acc /= (len_0 + len_1 + len_more)
# print("准确率为: {:.3f}".format(acc))

history = model.fit(x_train, y_train, epochs=10, batch_size=batchsz, validation_data=(x_test, y_test), verbose=2)

# %%
model.evaluate(x_test, y_test, batch_size=batchsz, verbose=2)
history_dict = history.history
print(history_dict.keys())
train_loss = history_dict['loss']
train_acc = history_dict['accuracy']
test_loss = history_dict['val_loss']
test_acc = history_dict['val_accuracy']

# %%
Epochs = range(1, 1+len(train_acc))
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(Epochs, train_loss, 'r', label='train_loss')
plt.plot(Epochs, test_loss, 'b', label='test_loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.show()

# %%
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(Epochs, train_acc, 'r', label='train_acc')
plt.plot(Epochs, test_acc, 'b', label='test_acc')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.show()
# %%
print("c")
