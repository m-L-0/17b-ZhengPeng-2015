import numpy as np
import mnist_reader
from tqdm import tqdm
from scipy import misc
import tensorflow as tf

np.random.seed(2017)
tf.set_random_seed(2017)

X_train, y_train = mnist_reader.load_mnist('../MNIST_data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('../MNIST_data/fashion', kind='t10k')

height,width = 128, 128

from keras.applications.mobilenet import MobileNet
from keras.layers import Input,Dense,Dropout,Lambda
from keras.models import Model
from keras import backend as K

input_image = Input(shape=(height,width))
input_image_ = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,3),3,3))(input_image)
base_model = MobileNet(input_shape=(height, width, 3), input_tensor=input_image_, include_top=False, pooling='avg')
output = Dropout(0.5)(base_model.output)
predict = Dense(10, activation='softmax')(output)

model = Model(inputs=input_image, outputs=predict)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

X_train = X_train.reshape((-1,28,28))
X_train = np.array([misc.imresize(x, (height,width)).astype(float) for x in tqdm(iter(X_train))])/255.

X_test = X_test.reshape((-1,28,28))
X_test = np.array([misc.imresize(x, (height,width)).astype(float) for x in tqdm(iter(X_test))])/255.

model.fit(X_train, y_train, batch_size=16, epochs=5, validation_data=(X_test, y_test))
