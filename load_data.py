# -*- coding: utf-8 -*-
"""
@author: HYPJUDY 2019/4/15
https://github.com/HYPJUDY

Decoupling Localization and Classification in Single Shot Temporal Action Detection
-----------------------------------------------------------------------------------

Functions to get train and test data

"""


import numpy as np
import random
from os.path import join
import pickle
from config import get_anno_ath, get_data_x_path
import sys

small_num_data_for_test = False


def read_window_info(path):
    result = []
    with open(path, 'r') as path:
        for line in path:
            sFrame, video_name = line.split(',')
            result.append([int(sFrame), video_name.strip()])
    return result


def read_pickle(path):
    with open(path, 'rb') as path:
        if sys.version_info[0] == 2:  # python2
            result = pickle.load(path)
        else:  # python3
            result = pickle.load(path, encoding='bytes')
    return result


############################# GET TRAIN DATA ##############################
# 由于一个batch可能存在多个gt标注，这里是将所有的gt遍历出来并记录下每个batch对应在新gt标注数组中的位置
def batch_data_process(batch_data):
    batch_size = len(batch_data) # 32
   
    new_batch_data = np.array(np.ones([1, batch_data[0].shape[1]])) 
    batch_start_index = [0]
    for i in range(batch_size):
        new_batch_data = np.concatenate((new_batch_data, batch_data[i]))
        if i < (batch_size - 1):
            batch_start_index.append(batch_start_index[-1] + len(batch_data[i]))
    new_batch_data = new_batch_data[1:]
    batch_start_index.append(len(new_batch_data))
    
    #返回由原先数据构成一个整体后的数据和每个子数据对应的其实索引
    return new_batch_data, np.array(batch_start_index)


def get_train_data(config, mode, pretrain_dataset, shuffle=True):
    batch_size = config.batch_size  # 32
    split_set = config.train_split_set # val
    data_x_path = get_data_x_path(config.feature_path, split_set, mode, pretrain_dataset)  # 获取特征文件路径

    anno_path = get_anno_ath(split_set)  # 获取标注文件路径
    # Since the dataX is matched with window_info.log,
    # window_info need to load from pre-defined file
    gt_label_file = join(anno_path, 'gt_label.pkl') # 获取gt_label.pkl路径
    gt_info_file = join(anno_path, 'gt_info.pkl') # 获取gt_info.pkl路径，其中保存了归一化后的窗口的信息
    
    gt_label = read_pickle(gt_label_file)
    gt_info = read_pickle(gt_info_file)

    if not small_num_data_for_test:
        num_data = len(gt_label)  # 4741
    else:
        num_data = batch_size

    # 存放每一个batch对应的数据
    batch_dataX = []
    batch_gt_label = []
    batch_gt_info = []
    batch_index = []

    
    batch_start_list = [i * batch_size for i in range(int(num_data / batch_size))]
    
    # 防止剩余的文件没有遍历到
    if (num_data - (batch_start_list[-1] + batch_size)) > (batch_size / 8):
        batch_start_list.append(num_data - batch_size)
    
    batch_shuffle_list = list(range(num_data))
    
    if shuffle:
        random.seed(6)
        random.shuffle(batch_shuffle_list)  # 打乱数据的顺序
        
    # 依次加载每一个batch的数据
    for bstart in batch_start_list:
        data_list = batch_shuffle_list[bstart:(bstart + batch_size)]  # 32个数据对应的索引 [424,1654,1621........]
        #定义三个列表保存一个batch的数据，包括特征数据和gt_label，gt_info
        tmp_batch_dataX = []
        tmp_batch_gt_label = []
        tmp_batch_gt_info = []
        #根据下标获取对应数据，最后追加到对应列表
        for idx in data_list:
            adataX = np.load(join(data_x_path, str(idx) + '.npy'))  
            tmp_batch_dataX.append(adataX)
            tmp_batch_gt_label.append(gt_label[idx]) 
            tmp_batch_gt_info.append(gt_info[idx]) 

        batch_dataX.append(np.array(tmp_batch_dataX))
        
        # 由于每个视频可能包含动作标注，这里就是将所有的标注去除，并生成新的起始索引
        tmp_batch_gt_label, start_index = batch_data_process(tmp_batch_gt_label)
        batch_gt_label.append(tmp_batch_gt_label)
        batch_index.append(start_index)

        tmp_batch_gt_info, start_index = batch_data_process(tmp_batch_gt_info)
        batch_gt_info.append(tmp_batch_gt_info)
        
    #返回获取到所有的batch的数据，以列表形式存放，列表中每一个元素都存储了一个batch的数据
    return batch_dataX, batch_gt_label, batch_gt_info, batch_index


############################# GET TEST DATA ##############################

def get_test_data(config, mode, pretrain_dataset):
    batch_size = config.batch_size
    split_set = config.test_split_set
    data_x_path = get_data_x_path(config.feature_path, split_set, mode, pretrain_dataset)
    anno_path = get_anno_ath(split_set)

    # Since the dataX is matched with window_info.log,
    # window_info need to load from pre-defined file
    # 在window_info.log中记录了每个窗口大小为512的窗口的起始帧，因此在测试的时候需要用到这个其实帧信息
    window_info_path = join(anno_path, 'window_info.log')
    window_info = read_window_info(window_info_path)

    if not small_num_data_for_test:
        num_data = len(window_info)
    else:
        num_data = batch_size

    batch_dataX = []
    batch_window_info = []

    batch_start_list = [i * batch_size for i in range(int(num_data / batch_size))]
    if (num_data - (batch_start_list[-1] + batch_size)) > (batch_size / 8):
        batch_start_list.append(num_data - batch_size)

    batch_list = list(range(num_data))

    for bstart in batch_start_list:
        data_list = batch_list[bstart:(bstart + batch_size)]
        tmp_batch_dataX = []
        tmp_batch_window_info = []
        for idx in data_list:
            adataX = np.load(join(data_x_path, str(idx) + '.npy'))
            tmp_batch_dataX.append(adataX)
            tmp_batch_window_info.append(window_info[idx])

        batch_dataX.append(np.array(tmp_batch_dataX))
        batch_window_info.append(tmp_batch_window_info)
    # 测试仅仅需要图片的数据和窗口信息
    return batch_dataX, batch_window_info


