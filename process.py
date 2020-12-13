#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/6 2:50 下午
# @Author  : zbl
# @Email   : blzhu0823@gmail.com
# @File    : process.py
# @Software: PyCharm
import math
import os
import argparse
import pickle
import random

import numpy as np
from sklearn.model_selection import train_test_split
from load_data import load_data
from augmentation import transform_image
from os.path import join as pjoin
from PIL import Image
import shutil
import tensorflow as tf


def read_img(img):
    np_img = np.array(img)
    return np_img


def cut(img, bbox):
    x0 = bbox['x0']
    y0 = bbox['y0']
    x1 = bbox['x1']
    y1 = bbox['y1']
    return img.crop((x0, y0, x1, y1))


def resize(img, size):
    return img.resize((size, size))


def diff(img1, img2):
    np_img1 = read_img(img1)
    np_img2 = read_img(img2)
    return np_img2 - np_img1


def process(img1, img2, bbox):
    img1 = cut(img1, bbox)
    img2 = cut(img2, bbox)
    diff_np_img = diff(resize(img1, args.size), resize(img2, args.size))
    return diff_np_img


def catagory(flaw_type, flaw_count):
    if os.path.exists(args.saved_data_path):
        shutil.rmtree(args.saved_data_path)
    if not os.path.exists(args.saved_data_path):
        os.mkdir(args.saved_data_path)
    for i in range(15):
        if not os.path.exists(pjoin(args.saved_data_path, 'type' + str(flaw_type[i]))):
            os.mkdir(pjoin(args.saved_data_path, 'type' + str(flaw_type[i])))

    datas = load_data(args)
    for data in datas:
        x = process(data['img1'], data['img2'], data['info']['bbox'])
        y = data['info']['flaw_type']
        Image.fromarray(x).save(pjoin(args.saved_data_path, 'type' + str(y),
                                      'pic' + str(flaw_count[flaw_type.index(y)]) + '.jpg'))
        flaw_count[flaw_type.index(y)] += 1


def aug_collection(flaw_type, flaw_count):
    collection1 = [24]
    # collection1 = [14]
    session = tf.Session()
    for type in collection1:
        flag = False
        times = math.ceil(600.0 / (flaw_count[flaw_type.index(type)] - 1) - 1)
        PATH = pjoin(args.saved_data_path, 'type' + str(type))
        paths = os.listdir(PATH)
        for path in paths:
            img = Image.open(pjoin(PATH, path))
            img.load()
            for time in range(times):
                img_array = np.array(img)
                transformed_img = transform_image(img_array)
                Image.fromarray(transformed_img.eval(session=session)).resize((args.size, args.size)).save(
                    pjoin(PATH, 'aug_pic' + str(flaw_count[flaw_type.index(type)]) + '.jpg'))
                if flaw_count[flaw_type.index(type)] == 600:
                    flag = True
                    break
                flaw_count[flaw_type.index(type)] += 1
            if flag:
                break
    session.close()


def split_data(flaw_type, file_path):
    train_data, test_data = [], []
    for type in flaw_type:
        paths = os.listdir(pjoin(args.saved_data_path, 'type' + str(type)))
        data = [[type, paths[i]] for i in range(len(paths))]
        train, test = train_test_split(data, test_size=args.test_ratio, random_state=42, shuffle=True)
        train_data += train
        test_data += test

    random.shuffle(train_data)
    random.shuffle(test_data)

    with open(pjoin(file_path, "train_data.txt"), "wb+") as f:
        pickle.dump(train_data, f)
    with open(pjoin(file_path, "test_data.txt"), "wb+") as f:
        pickle.dump(test_data, f)


def conv2numpy(flaw_type, file_path):
    with open(pjoin(file_path, "train_data.txt"), "rb") as f:
        train_data = pickle.load(f)
    with open(pjoin(file_path, "test_data.txt"), "rb") as f:
        test_data = pickle.load(f)

    x_train, y_train, x_test, y_test = [], [], [], []
    for i in train_data:
        img = Image.open(pjoin(args.saved_data_path, 'type' + str(i[0]), i[1]))
        x_train.append(np.expand_dims(np.array(img), axis=0))
        y_train.append(flaw_type.index(i[0]))
    for i in test_data:
        img = Image.open(pjoin(args.saved_data_path, 'type' + str(i[0]), i[1]))
        x_test.append(np.expand_dims(np.array(img), axis=0))
        y_test.append(flaw_type.index(i[0]))

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.vstack(y_train).squeeze()
    y_test = np.vstack(y_test).squeeze()

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    np.save(pjoin(file_path, 'x_train'), x_train)
    np.save(pjoin(file_path, 'x_test'), x_test)
    np.save(pjoin(file_path, 'y_train'), y_train)
    np.save(pjoin(file_path, 'y_test'), y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', default='D://fabric_data_new/', type=str)
    parser.add_argument('-saved_data_path', default='./processed_data', type=str)
    parser.add_argument('-size', default=50, type=int)
    parser.add_argument('-test_ratio', default=0.3, type=float)
    args = parser.parse_args()
    flaw_type = [1, 2, 3, 4, 5, 6, 7, 14, 16, 17, 20, 21, 22, 23, 24]
    # flaw_count = [1 for i in range(15)]

    # catagory(flaw_type, flaw_count)

    # aug_collection(flaw_type, flaw_count)

    split_data(flaw_type, "./info")

    conv2numpy(flaw_type, "./info" )

    # task3_selected = [1, 2, 5, 13]
    # X_task1 = []
    # X_task2 = []
    # X_task3 = []
    # Y_task1 = []
    # Y_task2 = []
    # Y_task3 = []
    # for data in datas:
    #     x = process(data['img1'], data['img2'], data['info']['bbox'], args.size)
    #     y = data['info']['flaw_type']
    #     X_task1.append(np.expand_dims(x, axis=0))
    #     X_task2.append(np.expand_dims(x, axis=0))
    #     Y_task1.append(y)
    #     if y >= 6 and y <= 12:
    #         Y_task2.append(6)
    #     elif y >= 9:
    #         Y_task2.append(y - 6)
    #     else:
    #         Y_task2.append(y)
    #     if y in task3_selected:
    #         X_task3.append(np.expand_dims(x, axis=0))
    #         Y_task3.append(task3_selected.index(y))
    #
    # X_task1 = np.vstack(X_task1)
    # X_task2 = np.vstack(X_task2)
    # X_task3 = np.vstack(X_task3)
    # Y_task1 = np.vstack(Y_task1).squeeze()
    # Y_task2 = np.vstack(Y_task2).squeeze()
    # Y_task3 = np.vstack(Y_task3).squeeze()
    #
    # if args.test_ratio is None:
    #     X_tasks = [X_task1, X_task2, X_task3]
    #     Y_tasks = [Y_task1, Y_task2, Y_task3]
    #     for i in range(3):
    #         if not os.path.exists(pjoin(args.saved_data_path, 'task' + str(i + 1))):
    #             os.mkdir(pjoin(args.saved_data_path, 'task' + str(i + 1)))
    #         np.save(pjoin(args.saved_data_path, 'task' + str(i + 1), 'X'), X_tasks[i])
    #         np.save(pjoin(args.saved_data_path, 'task' + str(i + 1), 'Y'), Y_tasks[i])
    #         print('X_train size for task' + str(i + 1) + ':', X_tasks[i].shape)
    #         print('Y_train size for task' + str(i + 1) + ':', Y_tasks[i].shape)
    #
    # else:
    #     X_tasks = [X_task1, X_task2, X_task3]
    #     Y_tasks = [Y_task1, Y_task2, Y_task3]
    #     for i in range(3):
    #         if not os.path.exists(pjoin(args.saved_data_path, 'task' + str(i + 1))):
    #             os.mkdir(pjoin(args.saved_data_path, 'task' + str(i + 1)))
    #         X_train, X_test, Y_train, Y_test = train_test_split(X_tasks[i], Y_tasks[i],
    #                                                             test_size=args.test_ratio,
    #                                                             random_state=42,
    #                                                             shuffle=True)
    #         np.save(pjoin(args.saved_data_path, 'task' + str(i + 1), 'X_train'), X_train)
    #         np.save(pjoin(args.saved_data_path, 'task' + str(i + 1), 'X_test'), X_test)
    #         np.save(pjoin(args.saved_data_path, 'task' + str(i + 1), 'Y_train'), Y_train)
    #         np.save(pjoin(args.saved_data_path, 'task' + str(i + 1), 'Y_test'), Y_test)
    #         print('X_train size for task' + str(i + 1) + ':', X_train.shape)
    #         print('X_test size for task' + str(i + 1) + ':', X_test.shape)
    #         print('Y_train size for task' + str(i + 1) + ':', Y_train.shape)
    #         print('Y_test size for task' + str(i + 1) + ':', Y_test.shape)
