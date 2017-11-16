import os
import shutil
import tensorflow as tf
import numpy as np
import time
import sys
import gzip


def get_dirs(file_path):
    """
    Description: Get file_names from a certain directory and
                 store them into a list.
    :param file_path: file_path
    :return: put all file_names in a list
    """
    if r'/' != file_path[-1]:
        file_path += r'/'
    dirs = []
    path_dir = os.listdir(file_path)
    for i in path_dir:
        child = os.path.join('%s%s' % (file_path, i))
        dirs.append(child)

    return dirs


def tfrecord2jpg(fashion_path):
    tfrecords_path = fashion_path + 'tfrecords/'
    try:
        train_file = gzip.GzipFile(tfrecords_path +
                                   "train.tfrecords.tar.gz", "r")
        with open(tfrecords_path + "train.tfrecords", 'wb') as f:
            f.write(train_file.read())
    except FileNotFoundError:
        pass
    tfrecords_dirs = sorted(get_dirs(tfrecords_path), key=len)
    for i in range(len(tfrecords_dirs)-1, -1, -1):
        if tfrecords_dirs[i].split('.')[-1] == "gz":
            tfrecords_dirs.pop(i)
    print('tfrecords_files to be transformed:', tfrecords_dirs)
    reader = tf.TFRecordReader()
    start_time = int(time.time())
    prev_time = start_time
    idx = 0

    for tfrecords_dir in tfrecords_dirs:
        suffix_path = tfrecords_dir.split(r'/')[-1].split('.')[0]
        filename_queue = tf.train.string_input_producer([tfrecords_dir])

        # 从 TFRecord 读取内容并保存到 serialized_example 中
        _, serialized_example = reader.read(filename_queue)
        # 读取 serialized_example 的格式
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # 解析从 serialized_example 读取到的内容
        images = tf.decode_raw(features['image_raw'], tf.uint8)
        labels = tf.cast(features['label'], tf.int64)

        print('Extracting {} has just started.'.format(tfrecords_dir))
        with tf.Session() as sess:
            # 启动多线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                step = 0
                while not coord.should_stop():
                    label, image = sess.run([labels, images])
                    image = np.fromstring(image, dtype=np.float32)
                    image = tf.reshape(image, [28, 28, 1])
                    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
                    image = tf.image.encode_jpeg(image)
                    with tf.gfile.GFile(fashion_path +
                                        'img_from_tfrecord/' +
                                        suffix_path +
                                        '/' +
                                        suffix_path +
                                        '_' +
                                        str(idx) +
                                        '_' +
                                        str(label) +
                                        '.jpg', 'wb') as f:
                        img = sess.run(image)
                        step += 1
                        f.write(img)
                    idx += 1
                    current_time = int(time.time())
                    lasting_time = current_time - start_time
                    interval_time = current_time - prev_time
                    if interval_time >= 1:
                        sys.stdout.flush()
                        sys.stdout.write("\rGenerating the {}-th image: {}, lasting {} seconds".format(
                                            idx,
                                            suffix_path +
                                            '_' + str(idx) +
                                            '_' + str(label) +
                                            '.jpg', lasting_time))
                        prev_time = current_time
            except tf.errors.OutOfRangeError as err:
                print('Extracting {} has been finished.'.format(
                                                        tfrecords_dir))
            finally:
                coord.request_stop()
            coord.join(threads)


def main():
    fashion_path = 'MNIST_data/fashion/'
    # get empty "img_from_tfrecord" directory
    for dtry in ['img_from_tfrecord/', 'img_from_tfrecord/test/',
                 'img_from_tfrecord/validation/', 'img_from_tfrecord/train/']:
        if os.path.isdir(fashion_path + dtry):
            if os.listdir(fashion_path + dtry):
                shutil.rmtree(fashion_path + dtry)
                os.mkdir(fashion_path + dtry)
        else:
            os.mkdir(fashion_path + dtry)

    tfrecord2jpg(fashion_path)


if __name__ == '__main__':                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    main()
