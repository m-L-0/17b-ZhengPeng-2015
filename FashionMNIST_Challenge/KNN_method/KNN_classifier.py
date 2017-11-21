import numpy as np
import tensorflow as tf
import pandas as pd
import gzip
import os


def load_mnist(path):
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets('../MNIST_data/fashion', one_hot=True)
    # read data
    import gzip
    import pandas as pd
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
    train_x, train_y, test_x, test_y = data
    train_x = train_x[:5000].reshape(train_x[:5000].shape[0], 784)
    train_y = train_y[:5000].reshape(train_y[:5000].shape[0])
    test_x = test_x[:200].reshape(test_x[:200].shape[0], 784)
    test_y = test_y[:200].reshape(test_y[:200].shape[0])

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
            print('train:', np.argmax(train_y[ansIndex]))
            print('test: ', np.argmax(test_y[i]))
            if np.argmax(test_y[i]) == np.argmax(train_y[ansIndex]):
                right += 1.0
        accracy = right/200.0
        print(accracy)


if __name__ == "__main__":
    path = r"../MNIST_data/fashion_csv/"
    mnist_data = load_mnist(path)
    knn(mnist_data)
