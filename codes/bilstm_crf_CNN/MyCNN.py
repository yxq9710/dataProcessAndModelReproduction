import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.constraints import max_norm

filters = 10
kernel_size = 5
batchsz = 128


class MyCNN(Model):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, padding='valid')
        self.relu = layers.ReLU()
        self.drop = layers.Dropout(rate=0.5)
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(16, activation='relu')
        self.d2 = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y


class_model = MyCNN()
class_model.build(input_shape=(batchsz, 28, 28, 1))
class_model.summary()


seq_model = Sequential([
    layers.Conv2D(filters, kernel_size, padding='valid', input_shape=(28, 28, 1)),
    layers.ReLU(),
    layers.Dropout(rate=0.5),
    layers.Flatten(),
    layers.Dense(16, activation='relu')
])
seq_model.add(layers.Dense(1, activation='sigmoid'))
seq_model.summary()  # 模型构建的三种方式：build, fit, input_shape=()
