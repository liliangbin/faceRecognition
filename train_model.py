# -*-coding:utf8-*-
from keras_preprocessing.image import ImageDataGenerator

__author__ = '万壑'

import numpy as np
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from dataSet import DataSet
from utils import show_train_history


# 建立一个基于CNN的人脸识别模型
class Model(object):
    FILE_PATH = "model.h5"  # 模型进行存储和读取的地方
    IMAGE_SIZE = 128  # 模型接受的人脸图片一定得是128*128的

    def __init__(self):
        self.model = None

    # 读取实例化后的DataSet类作为进行训练的数据源
    def read_trainData(self, dataset):
        self.dataset = dataset

    # 建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类
    def build_model(self):
        # self.model = Sequential()
        # self.model.add(
        #     Convolution2D(
        #         filters=32,
        #         kernel_size=(5, 5),
        #         padding='same',
        #         dim_ordering='th',
        #         input_shape=self.dataset.X_train.shape[1:]
        #     )
        # )
        #
        # self.model.add(Activation('relu'))
        # self.model.add(
        #     MaxPooling2D(
        #         pool_size=(2, 2),
        #         strides=(2, 2),
        #         padding='same'
        #     )
        # )
        #
        # self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        #
        # self.model.add(Flatten())
        # self.model.add(Dense(512))
        # self.model.add(Dropout(0.5))
        #
        # self.model.add(Activation('relu'))
        #
        # self.model.add(Dense(self.dataset.num_classes))
        # self.model.add(Activation('softmax'))

        self.model = Sequential()

        # 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering='th',
                                     input_shape=self.dataset.X_train.shape[1:]))  # 1 2维卷积层
        self.model.add(Activation('relu'))  # 2 激活函数层

        self.model.add(Convolution2D(32, 3, 3))  # 3 2维卷积层
        self.model.add(Activation('relu'))  # 4 激活函数层

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 5 池化层
        self.model.add(Dropout(0.25))  # 6 Dropout层

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))  # 7  2维卷积层
        self.model.add(Activation('relu'))  # 8  激活函数层

        self.model.add(Convolution2D(64, 3, 3))  # 9  2维卷积层
        self.model.add(Activation('relu'))  # 10 激活函数层

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 11 池化层
        self.model.add(Dropout(0.25))  # 12 Dropout层

        self.model.add(Flatten())  # 13 Flatten层
        self.model.add(Dense(512))  # 14 Dense层,又被称作全连接层
        self.model.add(Activation('relu'))  # 15 激活函数层
        self.model.add(Dropout(0.5))  # 16 Dropout层
        self.model.add(Dense(self.dataset.num_classes))  # 17 Dense层
        print(self.dataset.num_classes, "====>numclasses")
        self.model.add(Activation('softmax'))

        self.model.summary()

    # 进行模型训练的函数，具体的optimizer、loss可以进行不同选择
    def train_model(self):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)
        self.model.compile(
            optimizer=sgd,  # 有很多可选的optimizer，例如RMSprop,Adagrad，你也可以试试哪个好，我个人感觉差异不大
            loss='categorical_crossentropy',  # 你可以选用squared_hinge作为loss看看哪个好
            metrics=['accuracy'])

        datagen = ImageDataGenerator(
            featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
            samplewise_center=False,  # 是否使输入数据的每个样本均值为0
            featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
            samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
            zca_whitening=False,  # 是否对输入数据施以ZCA白化
            rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
            width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
            height_shift_range=0.2,  # 同上，只不过这里是垂直
            horizontal_flip=True,  # 是否进行随机水平翻转
            vertical_flip=False)  # 是否进行随机垂直翻转
        train_history = self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                              batch_size=20),
                                                 samples_per_epoch=dataset.train_images.shape[0],
                                                 nb_epoch=100,
                                                 validation_split=0.2)

    # epochs、batch_size为可调的参数，epochs为训练多少轮、batch_size为每次训练多少个样本
    # train_history = self.model.fit(self.dataset.X_train, self.dataset.Y_train, validation_split=0.2, epochs=25,
    #                                batch_size=20)
        show_train_history(train_history, 'acc', 'val_acc')


def evaluate_model(self):
    print('\nTesting---------------')
    loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)

    print('test loss;', loss)
    print('test accuracy:', accuracy)


def save(self, file_path=FILE_PATH):
    print('Model Saved.')
    self.model.save(file_path)


def load(self, file_path=FILE_PATH):
    print('Model Loaded.')
    self.model = load_model(file_path)


# 需要确保输入的img得是灰化之后（channel =1 )且 大小为IMAGE_SIZE的人脸图片
def predict(self, img):
    img = img.reshape((1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE))
    img = img.astype('float32')
    img = img / 255.0

    result = self.model.predict_proba(img)  # 测算一下该img属于某个label的概率
    max_index = np.argmax(result)  # 找出概率最高的

    return max_index, result[0][max_index]  # 第一个参数为概率最高的label的index,第二个参数为对应概率


if __name__ == '__main__':
    dataset = DataSet('./result/')
    model = Model()
    model.read_trainData(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()
