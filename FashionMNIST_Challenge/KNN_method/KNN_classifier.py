import numpy as np
import tensorflow as tf
import pandas as pd
import gzip
import os


def load_data(path):
    # read data from the path folder
    from sklearn.utils import shuffle
    train_file = gzip.GzipFile(path + "fashion-mnist_train.csv.tar.gz", "r")
    with open(path + "fashion-mnist_train.csv", 'wb') as f:
        f.write(train_file.read())
    data_train = pd.read_csv(path + 'fashion-mnist_train.csv')
    data_train = data_train.drop(index=len(data_train.iloc[:, 0])-1)
    os.remove(path + 'fashion-mnist_train.csv')
    data_test = pd.read_csv(path + 'fashion-mnist_test.csv')
    data_train = shuffle(np.array(data_train))
    data_test = shuffle(np.array(data_test))
    x_train = data_train[:, 1:]
    x_train = x_train.reshape(x_train.shape[0],
                              int(np.sqrt(x_train.shape[1])),
                              int(np.sqrt(x_train.shape[1])),
                              1)
    y_train = data_train[:, 0]
    x_test = data_test[:, 1:]
    x_test = x_test.reshape(x_test.shape[0],
                            int(np.sqrt(x_test.shape[1])),
                            int(np.sqrt(x_test.shape[1])),
                            1)
    y_test = data_test[:, 0]
    return x_train, y_train, x_test, y_test


def knn(data):
    # extract part of the data and do the transformation on the y data
    train_x = data[0][:].reshape(data[0][:].shape[0], 784)
    train_y = data[1][:].reshape(data[1][:].shape[0])
    test_x = data[2][:].reshape(data[2][:].shape[0], 784)
    test_y = data[3][:].reshape(data[3][:].shape[0])

    t_y = []
    for j in range(train_y.shape[0]):
        t = train_y[j]
        t_y.append(np.zeros((10)))
        t_y[j][int(t)] = 1
    train_y = np.array(t_y)
    t_y = []
    for j in range(test_y.shape[0]):
        t = test_y[j]
        t_y.append(np.zeros((10)))
        t_y[j][int(t)] = 1
    test_y = np.array(t_y)

    xtr = tf.placeholder(tf.float32, [None, 784])
    xte = tf.placeholder(tf.float32, [784])
    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(xtr, tf.negative(xte)), 2),
                                     reduction_indices=1))

    pred = tf.argmin(distance, 0)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        right = 0
        for i in range(200):
            ansIndex = sess.run(pred, {xtr: train_x, xte: test_x[i, :]})
            print('train:', np.argmax(train_y[ansIndex]), end=", ")
            print('test: ', np.argmax(test_y[i]))
            if np.argmax(test_y[i]) == np.argmax(train_y[ansIndex]):
                right += 1.0
        accracy = right / 200.0
        print("accracy: {}".format(accracy))


if __name__ == "__main__":
    path = r"../MNIST_data/fashion_csv/"
    mnist_data = load_data(path)
    knn(mnist_data)
