# %%
import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)

# %%
imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)  # 提取最常见的10000个词
print(x_train.shape, y_train.shape)
print(x_train[0])
print(len(x_train[0]), len(x_train[1]))

# %% convert index to word
max_len = 256
vocab_size = 10000
word_index = imdb.get_word_index()
# print(len(word_index))   88584
# print(word_index)
word_index = {k: (v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
# print(len(word_index))   88588

index_word = dict([(v, k) for k, v in word_index.items()])
# print(index_word)


def get_review(text):
    return ' '.join(index_word.get(i, '?') for i in text)
print(get_review(x_train[0]))


# %%  word process
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, value=word_index['<PAD>'], padding='post', maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, value=word_index['<PAD>'], padding='post', maxlen=max_len)
print(len(x_train[0]), len(x_train[1]))
print(len(x_test[0]), len(x_test[1]))
print(x_train[10], y_train[10])

# %% split train/dev/test and batch dataset
x_valid = x_train[:10000]
y_valid = y_train[:10000]
x_train = x_train[10000:]
y_train = y_train[10000:]
# train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000)
# valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).shuffle(1000)
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# print(train_ds)

# %% build model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()

# %% compile and fit
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# (x_train, y_train) = train_ds
history = model.fit(x_train, y_train, epochs=40, validation_data=(x_valid,y_valid), batch_size=512, verbose=1)  #fit返回对象history

# %% evaluate model
result = model.evaluate(x_test, y_test, batch_size=512, verbose=2)
print(result)

# %% print history
history_dict = history.history
print(history_dict.keys())
print(history_dict['loss'])
train_loss = history_dict['loss']
train_accuracy = history_dict['acc']
valid_loss = history_dict['val_loss']
valid_accuracy = history_dict['val_acc']

# %% plot the train loss and valid loss
epochs = range(1, len(train_loss)+1)
plt.figure()
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, valid_loss, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

# %% plot the Training and Validation accuracy
plt.figure()
plt.plot(epochs, train_accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, valid_accuracy, 'b', label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.show()

# %%  print learned embedding vector 
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)
print(weights)
# %%

