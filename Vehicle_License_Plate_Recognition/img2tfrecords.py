import numpy as np
import tensorflow as tf
import time
import os
import cv2
import matplotlib.pyplot as plt


# 图片存放位置
PATH_DES = r'data_tfrecords/integers_tfrecords/'
PATH_RES = r'data/integers/'

# 图片信息
IMG_HEIGHT = 227
IMG_WIDTH = 227
IMG_CHANNELS = 3
# NUM_TRAIN = 7000
NUM_VALIDARION = sum([len(os.listdir(PATH_RES + i))
                      for i in os.listdir(PATH_RES)]) // 4


# 读取图片
def read_images(path_res, path_des):
    imgs = []
    labels = []
    path_res_dirs = sorted(os.listdir(path_res))
    for i in path_res_dirs:
        paths_images = os.listdir(path_res + i)     # 本想排序的, 但是字符串排序效果不尽人意.
        t_lst = [''.join((path_res, i, '/', t)) for t in paths_images]
        paths_images = t_lst.copy()
        del t_lst
        for j in range(len(paths_images)):
            c = 0
            img_current = cv2.resize(cv2.imread(paths_images[j], 0), (16, 28))
            ret, img_current_threshed = cv2.threshold(img_current,
                                                      127, 255,
                                                      cv2.THRESH_OTSU)
            h, w = img_current_threshed.shape
            t_c = np.array([[img_current_threshed[0][0],
                             img_current_threshed[0, w-1]],
                            [img_current_threshed[h-1, 0],
                             img_current_threshed[h-1, w-1]]])
            c = sum([(t_c[0, 0]//255), (t_c[1, 1]//255),
                     (t_c[0, 1]//255), (t_c[1, 0]//255)])
            if_reverse = sum([sum(img_current_threshed[0, :] // 255),
                              sum(img_current_threshed[:, w-1] // 255),
                              sum(img_current_threshed[h-1, :] // 255),
                              sum(img_current_threshed[:, 0] // 255)])\
                / ((h + w) * 2 + 4) > 0.5
            # if c >= 1:
            #     img_current_threshed = 255 - img_current_threshed
            if c > 3 or (c > 1 and if_reverse):
                img_current_threshed = 255 - img_current_threshed
            # img_current_threshed = img_current
            label_current = paths_images[j].split("/")[-2]
            # if i == '2':
            #     fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            #     ax0, ax1 = ax.ravel()
            #     ax0.imshow(img_current, cmap="gray")
            #     ax1.imshow(img_current_threshed, cmap="gray")
            #     plt.title(c)
            #     # print([img_current_threshed[0][0],
            #     #        img_current_threshed[0, w-1],
            #     #        img_current_threshed[h-1, 0],
            #     #        img_current_threshed[h-1, w-1]])
            #     plt.show()
            imgs.append(img_current_threshed)
            labels.append(np.uint8(label_current))   # 这里注意, 目前的label仅为0-9
    imgs = np.array(imgs)
    labels = np.array(labels)
    return imgs, labels


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(images, labels, name):
    # 获取要转换为TFRecord文件的图片数目
    num = images.shape[0]
    print("num:", num)
    print("images.shape:", images.shape)
    # 输出TFRecord文件的文件名
    filename = name + '.tfrecords'
    print('Writting', filename)
    # 创建一个writer来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(num):
        # 将图像矩阵转化为一个字符串
        img_raw = images[i].tostring()
        # 将一个样例转化为Example Protocol Buffer，并将所有需要的信息写入数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(labels[i])),
            'image_raw': _bytes_feature(img_raw)}))
        # 将example写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()
    print('Writting End')


def main():
    print('reading images begin')
    start_time = time.time()
    train_images, train_labels = read_images(PATH_RES, PATH_DES)
    duration = time.time() - start_time
    print("reading images end , cost %d sec" % duration)

    # get validation
    # validation_images = train_images[:NUM_VALIDARION]
    # validation_labels = train_labels[:NUM_VALIDARION]
    # train_images = train_images[NUM_VALIDARION:]
    # train_labels = train_labels[NUM_VALIDARION:]

    train_images = train_images[:]
    train_labels = train_labels[:]

    # convert to tfrecords
    print('convert to tfrecords begin')
    start_time = time.time()
    convert(train_images, train_labels, 'train')
    # convert(validation_images, validation_labels, 'validation')
    duration = time.time() - start_time
    print('convert to tfrecords end , cost %d sec' % duration)


if __name__ == '__main__':
    main()
