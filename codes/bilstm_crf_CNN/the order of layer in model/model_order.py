import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

# data_augmentation = keras.Sequential([
#     layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(img_height, img_width, 3)),
#     layers.experimental.preprocessing.RandomRotation(0.1),
#     layers.experimental.preprocessing.RandomZoom(0.1),
# ])

model = Sequential([      # 在模型里面写input_shape， 而不是在build处
  # data_augmentation,
  #   layers.experimental.preprocessing.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    # layers.Dense(num_classes)
    layers.Dense(2, activation="softmax")
])

batches = 128
# model.build(input_shape=(batches, 28, 28, 1))
model.summary()