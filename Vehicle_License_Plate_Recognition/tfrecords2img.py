import os
import shutil
import tensorflow as tf
import time
import sys


# 图片存放位置
PATH_RES = [
    r'data_tfrecords/integers_tfrecords/integers.tfrecords',
    r'data_tfrecords/alphabets_tfrecords/alphabets.tfrecords',
    r'data_tfrecords/Chinese_letters_tfrecords/Chinese_letters.tfrecords'
    ]
PATH_DES = [
    r'imgs_from_tfrecords/integers/',
    r'imgs_from_tfrecords/alphabets/',
    r'imgs_from_tfrecords/Chinese_letters/'
    ]

PATH = list(zip(PATH_RES, PATH_DES))


def tfrecord2jpg(path_res, path_des):
    print('tfrecords_files to be transformed:', path_res)
    reader = tf.TFRecordReader()
    start_time = int(time.time())
    prev_time = start_time
    idx = 0

    filename_queue = tf.train.string_input_producer([path_res])

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

    print('Extracting {} has just started.'.format(path_res))
    with tf.Session() as sess:
        # 启动多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            flag_stop = 0
            label_count = set()
            while not coord.should_stop() or flag_stop:
                label, image = sess.run([labels, images])
                image = tf.reshape(image, [28, 16, 1])
                image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
                image = tf.image.encode_jpeg(image)
                if len(label_count) == 10 and label in label_count:
                    # prevent reading unlimitedly
                    flag_stop = 1
                    return 0
                    continue
                label_count |= {label}
                with tf.gfile.GFile(path_des +
                                    str(idx) +
                                    '_' +
                                    str(label) +
                                    '.jpg', 'wb') as f:
                    img = sess.run(image)
                    f.write(img)
                idx += 1
                current_time = int(time.time())
                lasting_time = current_time - start_time
                interval_time = current_time - prev_time
                if interval_time >= 1:
                    sys.stdout.flush()
                    sys.stdout.write("\rGenerating the {}-th image: {},\
                                     lasting {} seconds".format(
                                        idx,
                                        path_des +
                                        str(idx) + '_' +
                                        str(label) + '.jpg',
                                        lasting_time))
                    prev_time = current_time
        except tf.errors.OutOfRangeError as err:
            print('Extracting {} has been finished.'.format(
                                                    path_res))
        finally:
            coord.request_stop()
        coord.join(threads)


def main():
    # get empty directory
    for i in range(len(PATH)):
        if os.path.isdir(PATH_DES[i]):
            if os.listdir(PATH_DES[i]):
                shutil.rmtree(PATH_DES[i])
                os.mkdir(PATH_DES[i])
        else:
            os.mkdir(PATH_DES[i])
        tfrecord2jpg(PATH_RES[i], PATH_DES[i])


if __name__ == "__main__":
    main()
