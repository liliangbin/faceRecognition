# -*-coding:utf8-*-
__author__ = '万壑'

import cv2

from read_data import read_name_list, read_file
from train_model import Model


def onePicture(path):
    model = Model()
    model.load()
    img = cv2.imread(path)
    img = cv2.resize(img, (92, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    picType, prob = model.predict(img)
    if picType != -1:
        name_list = read_name_list('res')
        print(name_list)
        print(picType)
        print(name_list[picType], prob)
        print("done")
    else:
        print(" Don't know this person")


# 读取文件夹下子文件夹中所有图片进行识别
def Batch(path):
    model = Model()
    model.load()
    index = 0
    img_list, label_lsit, counter = read_file(path)
    for img in img_list:
        picType, prob = model.predict(img)
        if picType != -1:
            index += 1
            name_list = read_name_list('./res')
            print("done")
            print(name_list)
            print(picType)
            print(name_list[picType])
        else:
            print(" Don't know this person")

    return index


if __name__ == '__main__':
    onePicture('./test/chunmei.jpg')
    onePicture('./test/test.jpg')
    onePicture('./test/9.pgm')


    #Batch(".\\test")
