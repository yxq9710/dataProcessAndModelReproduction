from function_test import TextCNN
from tensorflow.keras.models import load_model
from dataset.imdb import data_process_imdb as dp

x_train, y_train, x_test, y_test, reverse_word_index = dp.load_imdb(num_words=10000)
x_train, x_test, max_len = dp.pad_sentence(x_train, x_test, max_len=64)

filepath = 'saved_model'
model = TextCNN(vocab_size=10000, output_dim=1, embedding_dim=100, max_len=max_len)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
text_cnn = load_model(filepath)
loss_trained, acc_trained = text_cnn.evaluate(x_test, y_test, verbose=2)
