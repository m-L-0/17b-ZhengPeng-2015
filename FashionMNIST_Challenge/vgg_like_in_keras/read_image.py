import numpy as np


class Dataset(object):
    def __init__(self, dtype='uint8', is_row_image=False):
        '''数据集
        
        Args:
            dtype: uint8 或 float32，uint8：每个像素值的范围是[0, 255];float32像素值范围是[0., 1.]
            is_row_image: 是否将3维图片展开成1维
        '''
        images = np.fromfile('./images/test_image.bin', dtype=np.uint8).reshape(-1, 28, 28, 1)
        print(images.shape)
        if dtype == 'uint8':
            self.images = images
        else:
            images = images.astype(np.float32) / 255.
            self.images = images
        if is_row_image:
            self.images = images.reshape([-1, 784])
        self.num_of_images = 6500
        self.offset = 0
        print('共6500张图片')

    def next_batch(self, batch_size=50):
        # 返回False表示以及没有样本
        # 注意：最后一个批次可能不足batch_size 所以推荐选择6500可以整除的batch_size
        if (self.offset + batch_size) <= self.num_of_images:
            self.offset += batch_size
            return self.images[self.offset-batch_size : self.offset]
        elif self.offset < self.num_of_images:
            return self.images[self.offset : ]
        else:
            False

if __name__ == '__main__':
    images = Dataset()
    b_img = images.next_batch()
    print(b_img.shape)
