import numpy as np
import tensorflow as tf
import time
import os
import cv2


# 图片存放位置
PATH_DES = [
    r'data_tfrecords/integers_tfrecords/integers.tfrecords',
    r'data_tfrecords/alphabets_tfrecords/alphabets.tfrecords',
    r'data_tfrecords/Chinese_letters_tfrecords/Chinese_letters.tfrecords'
    ]
PATH_RES = [r'data/integers/',
            r'data/alphabets/',
            r'data/Chinese_letters/']

PATH = list(zip(PATH_RES, PATH_DES))
# transformation between integer <-> string
integers = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9
}
alphabets = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'O': 14,
    'P': 15,
    'Q': 16,
    'R': 17,
    'S': 18,
    'T': 19,
    'U': 20,
    'V': 21,
    'W': 22,
    'X': 23,
    'Y': 24,
    'Z': 25
}
provinces = {
    '藏': 0,
    '川': 1,
    '鄂': 2,
    '甘': 3,
    '赣': 4,
    '广': 5,
    '桂': 6,
    '贵': 7,
    '黑': 8,
    '沪': 9,
    '吉': 10,
    '冀': 11,
    '津': 12,
    '晋': 13,
    '京': 14,
    '辽': 15,
    '鲁': 16,
    '蒙': 17,
    '闽': 18,
    '宁': 19,
    '青': 20,
    '琼': 21,
    '陕': 22,
    '苏': 23,
    '皖': 24,
    '湘': 25,
    '新': 26,
    '渝': 27,
    '豫': 28,
    '粤': 29,
    '云': 30,
    '浙': 31
}
label_ref = [
    integers,
    alphabets,
    provinces
]


# 图片信息
IMG_HEIGHT = 28
IMG_WIDTH = 16
IMG_CHANNELS = 1
# NUM_TRAIN = 7000
NUM_VALIDARION = [sum([len(os.listdir(r + i))
                       for i in os.listdir(r)]) // 4 for r in PATH_RES]


# 读取图片
def read_images(path_res, label_ref):
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
            if c > 2 or (c > 1 and if_reverse):
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
            imgs.append((img_current_threshed // 255).astype(np.uint8))
            labels.append(np.uint8(label_ref[label_current]))
    imgs = np.array(imgs)
    labels = np.array(labels)
    return imgs, labels


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(images, labels, filename):
    # 获取要转换为TFRecord文件的图片数目
    num = images.shape[0]
    print("num:", num)
    print("images.shape:", images.shape)
    # 输出TFRecord文件的文件名
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
    start_time = time.time()
    for i in range(len(PATH)):
        print('reading images from {} begin'.format(PATH_RES[i]))
        train_images, train_labels = read_images(PATH_RES[i], label_ref[i])
        # Slice data here.
        print('convert to tfrecords into {} begin'.format(PATH_DES[i]))
        convert(train_images, train_labels, PATH_DES[i])
    duration = time.time() - start_time
    print('Converting end , total cost = %d sec' % duration)


if __name__ == '__main__':
    main()
