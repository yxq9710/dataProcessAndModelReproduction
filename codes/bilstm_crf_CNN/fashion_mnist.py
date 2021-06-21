# %% import modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np


# %% load data and show data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_image, train_label), (test_image, test_label) = fashion_mnist.load_data()
print(type(train_image[1]))
print(train_image.shape)
print(test_image.shape)
print(train_label[2:10])
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_image[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_image = train_image/255.0
test_image = test_image/255.0

plt.figure(figsize=(5, 5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_image[i], cmap=plt.cm.binary)  # cmap参数使图像为黑白图像(binary)
    plt.xlabel(class_names[train_label[i]])
plt.show()

# %%  use tf.data to process
train_ds = tf.data.Dataset.from_tensor_slices((train_image, train_label)).shuffle(10000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((test_image, test_label)).batch(128)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = Flatten()   # 在模型的第一层要定义input_shape
        self.drop = Dropout(rate=0.5)  # 降低了模型的过拟合
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.drop(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


model = MyModel()

#%%
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


def train_step(image, label):
    with tf.GradientTape() as tape:
        predictions = model(image)
        loss = loss_object(label, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(label, predictions)


def _test_step(image, label):
    predictions = model(image)
    t_loss = loss_object(label, predictions)

    test_loss(t_loss)
    test_accuracy(label, predictions)


#%%
EPOCHS = 10
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    for train_image, train_label in train_ds:
        train_step(train_image, train_label)
    # train_step(train_image, train_label)
    # _test_step(test_image, test_label)
    for test_image, test_label in test_ds:
        _test_step(test_image, test_label)

    temp = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(temp.format(epoch+1,
                      train_loss.result(),
                      train_accuracy.result() * 100,
                      test_loss.result(),
                      test_accuracy.result() * 100))

# %% use model to predict
predictions = model.predict(test_image)
print(tf.argmax(predictions[0]))
