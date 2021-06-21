from dataset.imdb import data_process_imdb as dp
from cnn_cnn import CNN_CNN as BCNN
from dataset.ner_annotated_corpus.dataset.plt_loss import plot_loss_and_accuracy as pt

MAX_LEN = 64
vocab_size = 10000
batchsz = 256

x_train, y_train, x_test, y_test, reverse_word_index = dp.load_imdb(num_words=vocab_size)
x_train, x_test, max_len = dp.pad_sentence(x_train, x_test, max_len=MAX_LEN)


model = BCNN(vocab_size=vocab_size, max_len=MAX_LEN)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=batchsz,
                    validation_data=(x_test, y_test), verbose=1)    # batch_size默认为32
model.summary()
pt(history)



