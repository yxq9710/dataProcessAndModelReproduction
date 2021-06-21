from keras.models import load_model
from data_process_ner import processed_data, dataset
from bilstm_crf import Bulid_Bilstm_CRF
from keras.models import model_from_json
from keras.models import Sequential


x_train, y_train, x_test, y_test, max_len, vocab_size, num_class = dataset()
embedding_dim = 50
hidden_dim = 64
drop_rate = 0.3


filepath="ner-bi-lstm-td-model-0.86.hdf5"
model = Bulid_Bilstm_CRF(max_len, vocab_size, embedding_dim, hidden_dim, num_class, drop_rate)
model.load_weights(filepath)
loss, accuracy = model.evaluate(x_test, y_test)

