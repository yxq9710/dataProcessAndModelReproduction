import json
import numpy as np


def load_train_data(Epochs):
    with open('E:/sentiment_classification/codes/example_for_use_tf_CNN/train_0_' + str(Epochs) + '.json', 'r') as f:
        data0 = json.load(f)
    (train_x_0, train_y_0) = json.loads(data0)
    train_x_0 = np.array(train_x_0)
    train_y_0 = np.array(train_y_0)
    with open('E:/sentiment_classification/codes/example_for_use_tf_CNN/train_1_' + str(Epochs) + '.json', 'r') as f:
        data1 = json.load(f)
    (train_x_1, train_y_1) = json.loads(data1)
    with open('E:/sentiment_classification/codes/example_for_use_tf_CNN/train_more_' + str(Epochs) + '.json', 'r') as f:
        datamore = json.load(f)
    (train_x_more, train_y_more) = json.loads(datamore)
    train_x_1, train_y_1 = np.array(train_x_1), np.array(train_y_1)
    train_x_more, train_y_more = np.array(train_x_more), np.array(train_y_more)
    return train_x_0, train_y_0, train_x_1, train_y_1, train_x_more, train_y_more


def load_test_data(Epochs):
    with open('E:/sentiment_classification/codes/example_for_use_tf_CNN/test_0_' + str(Epochs) + '.json', 'r') as f:
        data0 = json.load(f)
    (test_x_0, test_y_0) = json.loads(data0)
    test_x_0 = np.array(test_x_0)
    test_y_0 = np.array(test_y_0)
    with open('E:/sentiment_classification/codes/example_for_use_tf_CNN/test_1_' + str(Epochs) + '.json', 'r') as f:
        data1 = json.load(f)
    (test_x_1, test_y_1) = json.loads(data1)
    with open('E:/sentiment_classification/codes/example_for_use_tf_CNN/test_more_' + str(Epochs) + '.json', 'r') as f:
        datamore = json.load(f)
    (test_x_more, test_y_more) = json.loads(datamore)
    test_x_1, test_y_1 = np.array(test_x_1), np.array(test_y_1)
    test_x_more, test_y_more = np.array(test_x_more), np.array(test_y_more)
    return test_x_0, test_y_0, test_x_1, test_y_1, test_x_more, test_y_more
