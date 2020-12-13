#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/6 3:09 下午
# @Author  : zbl
# @Email   : blzhu0823@gmail.com
# @File    : load_data.py
# @Software: PyCharm


import argparse
import os
import json
from os.path import join as pjoin
from PIL import Image
import tensorflow as tf


def load_data(args):
    datas = []
    exs = ['.jpg', '.json']
    data_path = args.data_path
    paths = os.listdir(pjoin(data_path, 'label_json'))
    labels_list = []
    flaw_dict = {1: [0, "逃花"], 2: [0, "塞网"], 3: [0, "破洞"], 4: [0, "缝头"], 5: [0, "水渍"], 6: [0, "脏污"], 7: [0, "白条"],
                 14: [0, "未对齐"], 16: [0, "伪色"], 17: [0, "前后色差"], 20: [0, "模板取错"], 21: [0, "漏浆"], 22: [0, "脱浆"],
                 23: [0, "色纱"], 24: [0, "飞絮"]}
    for path in paths:
        prefixs = list(map(lambda s: s.split('.')[0], os.listdir(pjoin(data_path, 'label_json', path))))
        for prefix in prefixs:
            try:
                img1 = Image.open(pjoin(data_path, 'temp', path, prefix + exs[0]))
                img2 = Image.open(pjoin(data_path, 'trgt', path, prefix + exs[0]))
                info = json.load(open(pjoin(data_path, 'label_json', path, prefix + exs[1]), encoding='utf-8'))
            except:
                continue
            if info["flaw_type"] in flaw_dict:
                datas.append({'img1': img1, 'img2': img2, 'info': info})
                flaw_dict[info["flaw_type"]][0] += 1
                labels_list.append(
                    "maxl: {:4d} flaw_type: {} location: {} x0: {} x1: {} y0: {} y1: {}\n".format(
                        max(info["bbox"]["x1"] - info["bbox"]["x0"], info["bbox"]["y1"] - info["bbox"]["y0"]),
                        info["flaw_type"],
                        pjoin(data_path, 'temp', path, prefix + exs[0]),
                        info["bbox"]["x0"], info["bbox"]["x1"],
                        info["bbox"]["y0"], info["bbox"]["y1"]))

    labels_list.sort(key=lambda s: s[6:10], reverse=True)
    collectedlabels = open("./info/label.txt", "w+", encoding="utf-8")
    collectedlabels.writelines("各类别样本数量\n")
    for key, value in flaw_dict.items():
        collectedlabels.writelines("{:2d}({}): {}\n".format(key, value[1], value[0]))
    collectedlabels.writelines("\n\n各样本信息\n")
    for i in labels_list:
        collectedlabels.writelines(i)
    collectedlabels.close()
    return datas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', default='D://fabric_data_new/', type=str)
    args = parser.parse_args()
    datas = load_data(args)
