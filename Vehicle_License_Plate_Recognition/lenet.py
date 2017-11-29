import tensorflow as tf
import tfrecords2array
import numpy as np


def lenet(char_classes):
    y_train = []
    x_train = []
    y_test = []
    x_test = []
    for char_class in char_classes:
        train_data = tfrecords2array.tfrecord2array(
            r"./data_tfrecords/" + char_class + "_tfrecords/train.tfrecords")
        test_data = tfrecords2array.tfrecord2array(
            r"./data_tfrecords/" + char_class + "_tfrecords/test.tfrecords")
        y_train.append(train_data[0])
        x_train.append(train_data[1])
        y_test.append(test_data[0])
        x_test.append(test_data[1])
    for i in [y_train, x_train, y_test, x_test]:
        for j in i:
            print(j.shape)
    y_train = np.vstack(y_train)
    x_train = np.vstack(x_train)
    y_test = np.vstack(y_test)
    x_test = np.vstack(x_test)

    class_num = y_test.shape[-1]

    print("x_train.shape=" + str(x_train.shape))
    print("x_test.shape=" + str(x_test.shape))
    sess = tf.InteractiveSession()

    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, class_num])
    # 把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽， 第4维代表图像通道数, 1表示黑白
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层：卷积层
    conv1_weights = tf.get_variable(
        "conv1_weights",
        [5, 5, 1, 32],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 过滤器大小为5*5, 当前层深度为1， 过滤器的深度为32
    conv1_biases = tf.get_variable("conv1_biases", [32],
                                   initializer=tf.constant_initializer(0.0))
    conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1],
                         padding='SAME')
    # 移动步长为1, 使用全0填充
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))     # 激活函数Relu去线性化

    # 第二层：最大池化层
    # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

    # 第三层：卷积层
    conv2_weights = tf.get_variable(
        "conv2_weights",
        [5, 5, 32, 64],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 过滤器大小为5*5, 当前层深度为32， 过滤器的深度为64
    conv2_biases = tf.get_variable(
        "conv2_biases", [64], initializer=tf.constant_initializer(0.0))
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1],
                         padding='SAME')
    # 移动步长为1, 使用全0填充
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层：最大池化层
    # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

    # 第五层：全连接层
    fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64, 1024],
                                  initializer=tf.truncated_normal_initializer(
                                  stddev=0.1))
    # 7*7*64=3136把前一层的输出变成特征向量
    fc1_biases = tf.get_variable(
        "fc1_biases", [1024], initializer=tf.constant_initializer(0.1))
    pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_biases)

    # 为了减少过拟合，加入Dropout层
    keep_prob = tf.placeholder(tf.float32)
    fc1_dropout = tf.nn.dropout(fc1, keep_prob)

    # 第六层：全连接层
    fc2_weights = tf.get_variable("fc2_weights", [1024, class_num],
                                  initializer=tf.truncated_normal_initializer(
                                  stddev=0.1))
    # 神经元节点数1024, 分类节点10
    fc2_biases = tf.get_variable(
        "fc2_biases", [class_num], initializer=tf.constant_initializer(0.1))
    fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases

    # 第七层：输出层
    # softmax
    y_conv = tf.nn.softmax(fc2)

    # 定义交叉熵损失函数
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                                  reduction_indices=[1]))

    # 选择优化器，并让优化器最小化损失函数/收敛, 反向传播
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真实值
    # 判断预测值y和真实值y_中最大数的索引是否一致，y的值为1-class_num概率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    # 用平均值来统计测试准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 开始训练
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    batch_size = 64
    print("Training steps=" + str(x_train.shape[0]*7 // batch_size))
    for i in range(0, x_train.shape[0]*7, batch_size):
        if (i % x_train.shape[0]) > ((i + batch_size) % x_train.shape[0]):
            x_data_train = np.vstack(
                (x_train[i % x_train.shape[0]:],
                 x_train[:(i+batch_size) % x_train.shape[0]]))
            y_data_train = np.vstack(
                (y_train[i % y_train.shape[0]:],
                 y_train[:(i+batch_size) % y_train.shape[0]]))
            x_data_test = np.vstack((x_test[i % x_test.shape[0]:],
                                    x_test[:(i+batch_size) % x_test.shape[0]]))
            y_data_test = np.vstack((y_test[i % y_test.shape[0]:],
                                    y_test[:(i+batch_size) % y_test.shape[0]]))
        else:
            x_data_train = x_train[
                i % x_train.shape[0]:(i+batch_size) % x_train.shape[0]]
            y_data_train = y_train[
                i % y_train.shape[0]:(i+batch_size) % y_train.shape[0]]
            x_data_test = x_test[
                i % x_test.shape[0]:(i+batch_size) % x_test.shape[0]]
            y_data_test = y_test[
                i % y_test.shape[0]:(i+batch_size) % y_test.shape[0]]
        if i % 640 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={x: x_data_train, y_: y_data_train, keep_prob: 1.0})
            test_accuracy = accuracy.eval(
                feed_dict={x: x_data_test, y_: y_data_test, keep_prob: 1.0})
            print("step {}, training accuracy={}, testing accuracy={}".format(
                i, train_accuracy, test_accuracy))
        train_step.run(feed_dict={
            x: x_data_train, y_: y_data_train, keep_prob: 0.5})
    saver.save(sess, './my_model/')

    batch_size_test = 64
    epoch_test = y_test.shape[0] // batch_size_test + 1
    sum = 0
    for i in range(epoch_test):
        if (i % x_test.shape[0]) > ((i + batch_size_test) % x_test.shape[0]):
            x_data_test = np.vstack((
                x_test[i % x_train.shape[0]:],
                x_test[:(i+batch_size_test) % x_test.shape[0]]))
            y_data_test = np.vstack((
                y_test[i % y_test.shape[0]:],
                y_test[:(i+batch_size_test) % y_test.shape[0]]))
        else:
            x_data_test = x_test[
                i % x_test.shape[0]:(i+batch_size_test) % x_test.shape[0]]
            y_data_test = y_test[
                i % y_test.shape[0]:(i+batch_size_test) % y_test.shape[0]]
        c = accuracy.eval(feed_dict={
            x: x_data_test, y_: y_data_test, keep_prob: 1.0})
        sum += c
        # print("test accuracy %g" %  c)
    print("test accuracy %g" % (sum / epoch_test))

    print("Finish!")


def main():
    # integers:         4679
    # alphabets:        9796
    # Chinese_letters:  3974
    train_lst = ['integers', 'alphabets']
    lenet(train_lst)


if __name__ == '__main__':
    main()
