import tensorflow as tf


def data_preprocessing(data):
    preprocessed_data = data
    return preprocessed_data


def do_the_batch_operations_on_tfrecords(path_file):
    fileNameQue = tf.train.string_input_producer([path_file])
    reader = tf.TFRecordReader()
    key, value = reader.read(fileNameQue)
    features = tf.parse_single_example(
        value,
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'image_raw': tf.FixedLenFeature([], tf.string)}
        )

    img = tf.decode_raw(features["image_raw"], tf.uint8)
    label = tf.cast(features["label"], tf.uint8)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:

        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while True:
            try:
                imgArr = sess.run(img)
                # TODO certain operations
                img_current = imgArr.reshape(28, 16)
                print("img_current.shape=", img_current.shape)
                # TODO END
            except:
                break

        coord.request_stop()
        coord.join(threads)


def main():
    PATH_FILE = r"./data_tfrecords/integers_tfrecords/integers.tfrecords"
    do_the_batch_operations_on_tfrecords(PATH_FILE)


if __name__ == '__main__':
    main()
