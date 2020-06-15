# -*- coding: utf-8 -*-

"""
@author: HYPJUDY 2019/4/15
https://github.com/HYPJUDY

Decoupling Localization and Classification in Single Shot Temporal Action Detection
-----------------------------------------------------------------------------------

Operations used by Decouple-SSAD

"""


import pandas as pd
import pandas
import numpy as np
import numpy
import os
import tensorflow as tf
from os.path import join


#################################### TRAIN & TEST #####################################

def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def jaccard_with_anchors(anchors_min, anchors_max, len_anchors, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """

    int_xmin = tf.maximum(anchors_min, box_min)
    int_xmax = tf.minimum(anchors_max, box_max)

    inter_len = tf.maximum(int_xmax - int_xmin, 0.)

    union_len = len_anchors - inter_len + box_max - box_min
    jaccard = tf.div(inter_len, union_len)
    return jaccard

# idx始终为0，b_glabels应该为[0,0,0,0,1,0,0,0,....0]这种形式
def loop_condition(idx, b_anchors_rx, b_anchors_rw, b_glabels, b_gbboxes,
                   b_match_x, b_match_w, b_match_labels, b_match_scores):
    #idx：0
    #如果存在gt,那么tf.shape(b_glabels)必然大于0，循环条件就会成立
    r = tf.less(idx, tf.shape(b_glabels))
    return r[0]

#该函数是最难理解的一个函数猜测就是为生成的一个子batch的anchor找到匹配的gt赋予其相应的标签用于训练
def loop_body(idx, b_anchors_rx, b_anchors_rw, b_glabels, b_gbboxes,
              b_match_x, b_match_w, b_match_labels, b_match_scores):
    # idx=0
    # b_anchors_rw,b_anchors_rx保存了预测的一个batch的anchor的中心点和宽度
    # b_glabels,b_gbboxes保存了真实的一个batch的标签和坐标值
    num_class = b_match_labels.get_shape().as_list()[-1]  # 21
    
    #gt的信息，用于计算IOU值
    label = b_glabels[idx][0:num_class]
    box_min = b_gbboxes[idx, 0]
    box_max = b_gbboxes[idx, 1]

    # ground truth
    box_x = (box_max + box_min) / 2
    box_w = (box_max - box_min)

    # predict
    anchors_min = b_anchors_rx - b_anchors_rw / 2
    anchors_max = b_anchors_rx + b_anchors_rw / 2

    len_anchors = anchors_max - anchors_min

    #计算预测的anchor和真实gt的交并比
    jaccards = jaccard_with_anchors(anchors_min, anchors_max, len_anchors, box_min, box_max) #（80，）

    # jaccards > b_match_scores > -0.5 & jaccards > matching_threshold
    
    #获取IOU>0.5提议对应的mask
    mask = tf.greater(jaccards, b_match_scores)
    matching_threshold = 0.5
    mask = tf.logical_and(mask, tf.greater(jaccards, matching_threshold))
    mask = tf.logical_and(mask, b_match_scores > -0.5)

    imask = tf.cast(mask, tf.int32)
    fmask = tf.cast(mask, tf.float32)
   

    #如果和gt足够接近那么直接将这些位置anchor的信息更新为gt的信息，其余位置不更新，用的是预测值
    b_match_x = fmask * box_x + (1 - fmask) * b_match_x  #(80,)
    b_match_w = fmask * box_w + (1 - fmask) * b_match_w  #(80,)

    #将gt_label拓展到和预测结果相同的维度
    ref_label = tf.zeros(tf.shape(b_match_labels), dtype=tf.int32)  #(80,21)
    ref_label = ref_label + label
    
    # 只保留IOU>0.5的提议对应的gt_label标注
    b_match_labels = tf.matmul(tf.diag(imask), ref_label) + tf.matmul(tf.diag(1 - imask), b_match_labels)  #(80,21)

    b_match_scores = tf.maximum(jaccards, b_match_scores)  #(80,)
    return [idx + 1, b_anchors_rx, b_anchors_rw, b_glabels, b_gbboxes,
            b_match_x, b_match_w, b_match_labels, b_match_scores]


def default_box(layer_steps, scale, a_ratios):
    # layer_steps : 16
    # scale : 1/16(0.0625)
    # a_ratios : [0.5, 0.75, 1, 1.5, 2]
    
    # width_set： 5种宽度
    width_set = [scale * ratio for ratio in a_ratios]
    
    # center_set : 16个中心点
    center_set = [1. / layer_steps * i + 0.5 / layer_steps for i in range(layer_steps)]
    width_default = []
    center_default = []
    for i in range(layer_steps):
        for j in range(len(a_ratios)):
            width_default.append(width_set[j])
            center_default.append(center_set[i])
    width_default = np.array(width_default)  # (80,)
    center_default = np.array(center_default) # (80,)
    return width_default, center_default


def anchor_box_adjust(anchors, config, layer_name, pre_rx=None, pre_rw=None):
    # anchors:[32,80,24]
    # layer_name: AL1
    if pre_rx == None:
        # num_anchors:16
        # scale : 1/16(0.0625)
        # aspect_ratios : [0.5,0.75,1,1.5,2]
        dboxes_w, dboxes_x = default_box(config.num_anchors[layer_name],
                                         config.scale[layer_name], config.aspect_ratios[layer_name])
    else:
        dboxes_x = pre_rx
        dboxes_w = pre_rw
    
    # 获取每个预测anchor的预测结果
    anchors_conf = anchors[:, :, -3]  # conf信息
    anchors_rx = anchors[:, :, -2]  # rx信息
    anchors_rw = anchors[:, :, -1]  # rw信息
    anchors_rx = anchors_rx * dboxes_w * 0.1 + dboxes_x   # 参数化的坐标回归(activate这里可能有问题)
    anchors_rw = tf.exp(0.1 * anchors_rw) * dboxes_w

    num_class = anchors.get_shape().as_list()[-1] - 3
    anchors_class = anchors[:, :, :num_class]  # [32,80,21]
    return anchors_class, anchors_conf, anchors_rx, anchors_rw  


# This function is mainly used for producing matched ground truth with
# each adjusted anchors after predicting one by one
# the matched ground truth may be positive/negative,
# the matched x,w,labels,scores all corresponding to this anchor

# 获取每个预设anchor对应的gt标签
def anchor_bboxes_encode(anchors, glabels, gbboxes, Index, config, layer_name, pre_rx=None, pre_rw=None):
    num_anchors = config.num_anchors[layer_name] # (16,8,4)
    num_dbox = config.num_dbox[layer_name] # 5 
    num_classes = anchors.get_shape().as_list()[-1] - 3  # 21 

    dtype = tf.float32

    # 获取预设anchor及网络的预测结果
    anchors_class, anchors_conf, anchors_rx, anchors_rw = \
        anchor_box_adjust(anchors, config, layer_name, pre_rx, pre_rw)

    # 保存所有batch的结果
    batch_match_x = tf.reshape(tf.constant([]), [-1, num_anchors * num_dbox])       # (B,80)
    batch_match_w = tf.reshape(tf.constant([]), [-1, num_anchors * num_dbox])       # (B,80)
    batch_match_scores = tf.reshape(tf.constant([]), [-1, num_anchors * num_dbox])  # (B,80)
    batch_match_labels = tf.reshape(tf.constant([], dtype=tf.int32),
                                    [-1, num_anchors * num_dbox, num_classes])      # (B,80,21)

    for i in range(config.batch_size):
        shape = (num_anchors * num_dbox)
        
        # x,w,score,label
        match_x = tf.zeros(shape, dtype)   #(80,)
        match_w = tf.zeros(shape, dtype)   #(80,)
        match_scores = tf.zeros(shape, dtype)   #(80,)
        match_labels_other = tf.ones((num_anchors * num_dbox, 1), dtype=tf.int32)
        match_labels_class = tf.zeros((num_anchors * num_dbox, num_classes - 1), dtype=tf.int32)
        match_labels = tf.concat([match_labels_other, match_labels_class], axis=-1)  #(80,21)

        #预测某一个batch中的num_anchors*num_dbox的anchor的中心点和宽度
        #假如batch_size = 32,那么这一步获取32个中某一个的信息（80个anchor）
        b_anchors_rx = anchors_rx[i]
        b_anchors_rw = anchors_rw[i]
        b_glabels = glabels[Index[i]:Index[i + 1]]
        b_gbboxes = gbboxes[Index[i]:Index[i + 1]]

        # idx指示当前是哪一个gt_label,因为一个视频可能有多个gt_label
        idx = 0
        [idx, b_anchors_rx, b_anchors_rw, b_glabels, b_gbboxes,
         match_x, match_w, match_labels, match_scores] = \
            tf.while_loop(loop_condition, loop_body,
                          [idx, b_anchors_rx, b_anchors_rw,
                           b_glabels, b_gbboxes,
                           match_x, match_w, match_labels, match_scores])

        match_x = tf.reshape(match_x, [-1, num_anchors * num_dbox])  #(80,)-->(1,80)
        batch_match_x = tf.concat([batch_match_x, match_x], axis=0)  # 按batch维度拼接  (B,80)

        match_w = tf.reshape(match_w, [-1, num_anchors * num_dbox])  #(80,)-->(1,80)
        batch_match_w = tf.concat([batch_match_w, match_w], axis=0)

        match_scores = tf.reshape(match_scores, [-1, num_anchors * num_dbox])  #(80,)-->(1,80)
        batch_match_scores = tf.concat([batch_match_scores, match_scores], axis=0)

        match_labels = tf.reshape(match_labels, [-1, num_anchors * num_dbox, num_classes])  #(80,21)-->(1,80,21)
        batch_match_labels = tf.concat([batch_match_labels, match_labels], axis=0)

    ##anchors_class, anchors_conf, anchors_rx, anchors_rw中保存的是所有的预测结果
    ##batch_match_x, batch_match_w, batch_match_labels, batch_match_scores中保存的所有的anchor和gt的匹配情况
    
    ###最终获得的结果是batch_size的预测结果和匹配情况
    return [batch_match_x, batch_match_w, batch_match_labels, batch_match_scores,
            anchors_class, anchors_conf, anchors_rx, anchors_rw]


def in_conv(layer, initer=tf.contrib.layers.xavier_initializer(seed=5)):
    net = tf.layers.conv1d(inputs=layer, filters=1024, kernel_size=3, strides=1, padding='same',
                           activation=tf.nn.relu, kernel_initializer=initer)
    out = tf.layers.conv1d(inputs=net, filters=1024, kernel_size=3, strides=1, padding='same',
                           activation=None, kernel_initializer=initer)
    return out


def out_conv(layer, initer=tf.contrib.layers.xavier_initializer(seed=5)):
    net = tf.nn.relu(layer)
    out = tf.layers.conv1d(inputs=net, filters=1024, kernel_size=3, strides=1, padding='same',
                           activation=tf.nn.relu, kernel_initializer=initer)
    return out


############################ TRAIN and TEST NETWORK LAYER ###############################

def get_trainable_variables():
    trainable_variables_scope = [a.name for a in tf.trainable_variables()]
    trainable_variables_list = tf.trainable_variables()
    trainable_variables = []
    for i in range(len(trainable_variables_scope)):
        if ("base_feature_network" in trainable_variables_scope[i]) or \
                ("anchor_layer" in trainable_variables_scope[i]) or \
                ("predict_layer" in trainable_variables_scope[i]):
            trainable_variables.append(trainable_variables_list[i])
    return trainable_variables

######第一步 定义有关网络 包括：基本特征提取网络，main_anchor_layer,分类和提议网络#######

def base_feature_network(X, mode=''):
    # main network
    initer = tf.contrib.layers.xavier_initializer(seed=5)
    with tf.variable_scope("base_feature_network" + mode):
        # ----------------------- Base layers ----------------------
        net = tf.layers.conv1d(inputs=X, filters=512, kernel_size=9, strides=1, padding='same', # [bs,128,1024]--->[bs,128,512]
                               activation=tf.nn.relu, kernel_initializer=initer)
        
        net = tf.layers.max_pooling1d(inputs=net, pool_size=4, strides=2, padding='same')   # [bs,64,512]
      
        net = tf.layers.conv1d(inputs=net, filters=512, kernel_size=9, strides=1, padding='same',
                               activation=tf.nn.relu, kernel_initializer=initer)
       
        net = tf.layers.max_pooling1d(inputs=net, pool_size=4, strides=2, padding='same')  # [bs,32,512]
       
    return net


def main_anchor_layer(net, mode=''):
    # main network
    initer = tf.contrib.layers.xavier_initializer(seed=5)
    with tf.variable_scope("main_anchor_layer" + mode):
        # ----------------------- Anchor layers ----------------------
        MAL1 = tf.layers.conv1d(inputs=net, filters=1024, kernel_size=3, strides=2, padding='same',
                                activation=tf.nn.relu, kernel_initializer=initer)
        # tensorflow的数据输入维度为[N,H,W,C]
        # conv3d输入shape为：[batch, in_depth, in_height, in_width, in_channels]
        # [batch_size, 16, 1024]
        MAL2 = tf.layers.conv1d(inputs=MAL1, filters=1024, kernel_size=3, strides=2, padding='same',
                                activation=tf.nn.relu, kernel_initializer=initer)
        # [batch_size, 8, 1024]
        MAL3 = tf.layers.conv1d(inputs=MAL2, filters=1024, kernel_size=3, strides=2, padding='same',
                                activation=tf.nn.relu, kernel_initializer=initer)
        # [batch_size, 4, 1024]

    return MAL1, MAL2, MAL3


def branch_anchor_layer(MALs, name=''):
    MAL1, MAL2, MAL3 = MALs
    with tf.variable_scope("branch_anchor_layer" + name):
        BAL3 = out_conv(in_conv(MAL3))  # [batch_size, 4, 1024]

        #是不是因为TensorFlow中没有一维的反卷积所以才不得已用2d卷积
        BAL3_expd = tf.expand_dims(BAL3, 1)  # [batch_size, 1, 4, 1024]
        BAL3_de = tf.layers.conv2d_transpose(BAL3_expd, 1024, kernel_size=(1, 4),
                                             strides=(1, 2), padding='same')  # [batch_size, 1, 8, 1024]
        BAL3_up = tf.reduce_sum(BAL3_de, [1])  # [batch_size, 8, 1024]
        MAL2_in_conv = in_conv(MAL2)
        BAL2 = out_conv((MAL2_in_conv * 2 + BAL3_up) / 3)  # [batch_size, 8, 1024]

        MAL2_expd = tf.expand_dims(BAL2, 1)  # [batch_size, 1, 8, 1024]
        MAL2_de = tf.layers.conv2d_transpose(MAL2_expd, 1024, kernel_size=(1, 4),
                                             strides=(1, 2), padding='same')  # [batch_size, 1, 16, 1024]
        MAL2_up = tf.reduce_sum(MAL2_de, [1])  # [batch_size, 16, 1024]
        MAL1_in_conv = in_conv(MAL1)
        BAL1 = out_conv((MAL1_in_conv * 2 + MAL2_up) / 3)  # [batch_size, 16, 1024]

    return BAL1, BAL2, BAL3


# action or not + conf + location (center&width)
# Anchor Binary Classification and Regression
def biClsReg_predict_layer(config, layer, layer_name, specific_layer):
    num_dbox = config.num_dbox[layer_name]
    with tf.variable_scope("biClsReg_predict_layer" + layer_name + specific_layer):
        anchor = tf.layers.conv1d(inputs=layer, filters=num_dbox * (1 + 3),
                                  kernel_size=3, padding='same', kernel_initializer=
                                      tf.contrib.layers.xavier_initializer(seed=5))
        anchor = tf.reshape(anchor, [config.batch_size, -1, (1 + 3)])
    return anchor


# action or not + class score + conf + location (center&width)
# Action Multi-Class Classification and Regression
def mulClsReg_predict_layer(config, layer, layer_name, specific_layer):
    num_dbox = config.num_dbox[layer_name]
    ncls = config.num_classes
    with tf.variable_scope("mulClsReg_predict_layer" + layer_name + specific_layer):
        anchor = tf.layers.conv1d(inputs=layer, filters=num_dbox * (ncls + 3),
                                  kernel_size=3, padding='same', kernel_initializer=
                                      tf.contrib.layers.xavier_initializer(seed=5))
        anchor = tf.reshape(anchor, [config.batch_size, -1, (ncls + 3)])
    return anchor


#################################### TRAIN LOSS #####################################

def loss_function(anchors_class, anchors_conf, anchors_xmin, anchors_xmax,
                  match_x, match_w, match_labels, match_scores, config):
    match_xmin = match_x - match_w / 2
    match_xmax = match_x + match_w / 2

    #######正样本
    pmask = tf.cast(match_scores > 0.5, dtype=tf.float32)
    num_positive = tf.reduce_sum(pmask)
    num_entries = tf.cast(tf.size(match_scores), dtype=tf.float32)

    #######hard样本(得分低但是重叠度却高)
    hmask = match_scores < 0.5
    hmask = tf.logical_and(hmask, anchors_conf > 0.5)  # conf是指的重叠度
    hmask = tf.cast(hmask, dtype=tf.float32)
    num_hard = tf.reduce_sum(hmask)

    # the meaning of r_negative: the ratio of anchors need to choose from easy negative anchors
    # If we have `num_positive` positive anchors in training data,
    # then we only need `config.negative_ratio*num_positive` negative anchors
    # r_negative=(number of easy negative anchors need to choose from all easy negative) / (number of easy negative)
    # the meaning of easy negative: all-pos-hard_neg
    # 这里是事先假定了
    #----------------------------------------------------------------------------------------------------------------------------
    #               (1-num_hard/num_positive)*num_positive        (num_positive-num_hard)       (negative_nums-num_hard)
    #r_negative = ------------------------------------------ =   --------------------------- = ---------------------------
    #                (total_anchor-num_positive-num_hard)              num_negative                  num_negative（就是易分的负样本）
    #-----------------------------------------------------------------------------------------------------------------------------
    
    r_negative = (config.negative_ratio - num_hard / num_positive) * num_positive / (
            num_entries - num_positive - num_hard)  
    r_negative = tf.minimum(r_negative, 1)
    
    # 获取nmask
    nmask = tf.random_uniform(tf.shape(pmask), dtype=tf.float32)  # 注意nmask是随机生成的服从均匀分布
    nmask = nmask * (1. - pmask)   # 乘以(1-pmask)和(1-hmask)就可以避免选中这些位置
    nmask = nmask * (1. - hmask)
    nmask = tf.cast(nmask > (1. - r_negative), dtype=tf.float32)

    # class_loss
    weights = pmask + nmask + hmask
    class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=anchors_class, labels=match_labels)
    class_loss = tf.losses.compute_weighted_loss(class_loss, weights)

    # correct_pred = tf.equal(tf.argmax(anchors_class, 2), tf.argmax(match_labels, 2))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    # loc_loss
    weights = pmask
    loc_loss = abs_smooth(anchors_xmin - match_xmin) + abs_smooth(anchors_xmax - match_xmax)
    loc_loss = tf.losses.compute_weighted_loss(loc_loss, weights)

    # conf loss
    weights = pmask + nmask + hmask
    # match_scores is from jaccard_with_anchors
    conf_loss = abs_smooth(match_scores - anchors_conf)
    conf_loss = tf.losses.compute_weighted_loss(conf_loss, weights)

    return class_loss, loc_loss, conf_loss


#################################### POST PROCESS #####################################
# min_max_norm就是个sigmod函数,最终的分类得分为分类得分*置信度
def min_max_norm(X):
    # map [0,1] -> [0.5,0.73] (almost linearly) ([-1, 0] -> [0.26, 0.5])
    return 1.0 / (1.0 + np.exp(-1.0 * X))

# 处理每一个视频对应的预测结果
def post_process(df, config):
    # 每个类别对应的分类得分结果
    #---------------------------------------
    # 第0类： [0,00001,0.0004,.......0,00008](每个anchor对应该类的结果)
    # 第1类： [0,00001,0.0004,.......0,00008](每个anchor对应该类的结果)
    # 第2类： [0,00001,0.0004,.......0,00008](每个anchor对应该类的结果)
    # .。。。。。。。。。。。。。。。。。。。。。
    #----------------------------------------
    
    
    class_scores_class = [(df['score_' + str(i)]).values[:].tolist() for i in range(21)]
    # #---------------------------------------
    # 第1个anchor： [0,00001,0.0004,.......0,00008](每一个类别的结果)
    # 第2个anchor： [0,00001,0.0004,.......0,00008](每一个类别的结果)
    # 第3个anchor： [0,00001,0.0004,.......0,00008](每一个类别的结果)
    # .。。。。。。。。。。。。。。。。。。。。。
    #----------------------------------------
    class_scores_seg = [[class_scores_class[j][i] for j in range(21)] for i in range(len(df))]

    class_real = [0] + config.class_real  # num_classes + 1

    # save the top 2 or 3 score element
    # append the largest score element
    # 首先选取得分最高的结果，然后将最高的得分置为0后选取第二高的得分，以此类推选取前三种最高的得分
    class_type_list = []
    class_score_list = []
    for i in range(len(df)):
        class_score = np.array(class_scores_seg[i][1:]) * min_max_norm(df.conf.values[i])  # 最终的得分是分类得分乘以归一化的conf
        class_score = class_score.tolist()
        class_type = class_real[class_score.index(max(class_score)) + 1]  # 选取得分最高的作为类别
        class_type_list.append(class_type)
        class_score_list.append(max(class_score))

    resultDf1 = pd.DataFrame()
    resultDf1['out_type'] = class_type_list
    resultDf1['out_score'] = class_score_list
    resultDf1['start'] = df.xmin.values[:]
    resultDf1['end'] = df.xmax.values[:]

    # append the second largest score element
    class_type_list = []
    class_score_list = []
    for i in range(len(df)):
        class_score = np.array(class_scores_seg[i][1:]) * min_max_norm(df.conf.values[i])
        class_score = class_score.tolist()
        class_score[class_score.index(max(class_score))] = 0
        class_type = class_real[class_score.index(max(class_score)) + 1]
        class_type_list.append(class_type)
        class_score_list.append(max(class_score))

    resultDf2 = pd.DataFrame()
    resultDf2['out_type'] = class_type_list
    resultDf2['out_score'] = class_score_list
    resultDf2['start'] = df.xmin.values[:]
    resultDf2['end'] = df.xmax.values[:]
    resultDf1 = pd.concat([resultDf1, resultDf2])

    # # append the third largest score element (improve little and slow)
    class_type_list = []
    class_score_list = []
    for i in range(len(df)):
        class_score = np.array(class_scores_seg[i][1:]) * min_max_norm(df.conf.values[i])
        class_score = class_score.tolist()
        class_score[class_score.index(max(class_score))] = 0
        class_score[class_score.index(max(class_score))] = 0
        class_type = class_real[class_score.index(max(class_score)) + 1]
        class_type_list.append(class_type)
        class_score_list.append(max(class_score))

    resultDf2 = pd.DataFrame()
    resultDf2['out_type'] = class_type_list
    resultDf2['out_score'] = class_score_list
    resultDf2['start'] = df.xmin.values[:]
    resultDf2['end'] = df.xmax.values[:]
    resultDf1 = pd.concat([resultDf1, resultDf2])

    # resultDf1=resultDf1[resultDf1.out_score>0.05]

    resultDf1['video_name'] = [df['video_name'].values[0] for _ in range(len(resultDf1))]  # 最终输入到txt文件中的是(video_name,start,end,type,score)
    return resultDf1


def temporal_nms(config, dfNMS, filename, videoname):
    nms_threshold = config.nms_threshold
    fo = open(filename, 'a')

    typeSet = list(set(dfNMS.out_type.values[:]))  # 获取某个视频的所有预测类别
    for t in typeSet:
        tdf = dfNMS[dfNMS.out_type == t]

        t1 = np.array(tdf.start.values[:])
        t2 = np.array(tdf.end.values[:])
        scores = np.array(tdf.out_score.values[:])
        ttype = list(tdf.out_type.values[:])

        durations = t2 - t1
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            tt1 = np.maximum(t1[i], t1[order[1:]])
            tt2 = np.minimum(t2[i], t2[order[1:]])
            intersection = tt2 - tt1
            IoU = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

            inds = np.where(IoU <= nms_threshold)[0]
            order = order[inds + 1]

        for idx in keep:
            # class_real: do not have class 0 (ambiguous) -> remove all ambiguous class
            if ttype[idx] in config.class_real:
                if videoname in ["video_test_0001255", "video_test_0001058",
                                 "video_test_0001459", "video_test_0001195", "video_test_0000950"]:  # 25fps
                    strout = "%s\t%.3f\t%.3f\t%d\t%.4f\n" % (videoname, float(t1[idx]) / 25,
                                                             float(t2[idx]) / 25, ttype[idx], scores[idx])
                elif videoname == "video_test_0001207":  # 24fps
                    strout = "%s\t%.3f\t%.3f\t%d\t%.4f\n" % (videoname, float(t1[idx]) / 24,
                                                             float(t2[idx]) / 24, ttype[idx], scores[idx])
                else:  # most videos are 30fps
                    strout = "%s\t%.3f\t%.3f\t%d\t%.4f\n" % (videoname, float(t1[idx]) / 30,
                                                             float(t2[idx]) / 30, ttype[idx], scores[idx])
                fo.write(strout)


def fuse_two_stream(spatial_path, temporal_path):
    temporal_df = pd.read_csv(temporal_path)
    spatial_df = pd.read_csv(spatial_path)
    out_df = temporal_df
    out_df['conf'] = temporal_df.conf.values[:] * 2 / 3 + spatial_df.conf.values * 1 / 3
    out_df['xmin'] = temporal_df.xmin.values[:] * 2 / 3 + spatial_df.xmin.values * 1 / 3
    out_df['xmax'] = temporal_df.xmax.values[:] * 2 / 3 + spatial_df.xmax.values * 1 / 3
    out_df['score_0'] = temporal_df.score_0.values[:] * 2 / 3 + spatial_df.score_0.values * 1 / 3
    out_df['score_1'] = temporal_df.score_1.values[:] * 2 / 3 + spatial_df.score_1.values * 1 / 3
    out_df['score_2'] = temporal_df.score_2.values[:] * 2 / 3 + spatial_df.score_2.values * 1 / 3
    out_df['score_3'] = temporal_df.score_3.values[:] * 2 / 3 + spatial_df.score_3.values * 1 / 3
    out_df['score_4'] = temporal_df.score_4.values[:] * 2 / 3 + spatial_df.score_4.values * 1 / 3
    out_df['score_5'] = temporal_df.score_5.values[:] * 2 / 3 + spatial_df.score_5.values * 1 / 3
    out_df['score_6'] = temporal_df.score_6.values[:] * 2 / 3 + spatial_df.score_6.values * 1 / 3
    out_df['score_7'] = temporal_df.score_7.values[:] * 2 / 3 + spatial_df.score_7.values * 1 / 3
    out_df['score_8'] = temporal_df.score_8.values[:] * 2 / 3 + spatial_df.score_8.values * 1 / 3
    out_df['score_9'] = temporal_df.score_9.values[:] * 2 / 3 + spatial_df.score_9.values * 1 / 3
    out_df['score_10'] = temporal_df.score_10.values[:] * 2 / 3 + spatial_df.score_10.values * 1 / 3
    out_df['score_11'] = temporal_df.score_11.values[:] * 2 / 3 + spatial_df.score_11.values * 1 / 3
    out_df['score_12'] = temporal_df.score_12.values[:] * 2 / 3 + spatial_df.score_12.values * 1 / 3
    out_df['score_13'] = temporal_df.score_13.values[:] * 2 / 3 + spatial_df.score_13.values * 1 / 3
    out_df['score_14'] = temporal_df.score_14.values[:] * 2 / 3 + spatial_df.score_14.values * 1 / 3
    out_df['score_15'] = temporal_df.score_15.values[:] * 2 / 3 + spatial_df.score_15.values * 1 / 3
    out_df['score_16'] = temporal_df.score_16.values[:] * 2 / 3 + spatial_df.score_16.values * 1 / 3
    out_df['score_17'] = temporal_df.score_17.values[:] * 2 / 3 + spatial_df.score_17.values * 1 / 3
    out_df['score_18'] = temporal_df.score_18.values[:] * 2 / 3 + spatial_df.score_18.values * 1 / 3
    out_df['score_19'] = temporal_df.score_19.values[:] * 2 / 3 + spatial_df.score_19.values * 1 / 3
    out_df['score_20'] = temporal_df.score_20.values[:] * 2 / 3 + spatial_df.score_20.values * 1 / 3

    out_df = out_df[out_df.score_0 < 0.99]
    # outDf.to_csv(fusePath, index=False)
    return out_df


def result_process(batch_win_info, batch_result_class,
                   batch_result_conf, batch_result_xmin, batch_result_xmax, config, batch_idx):
    out_df = pandas.DataFrame(columns=config.outdf_columns)
    for j in range(config.batch_size):
        tmp_df = pandas.DataFrame()
        # ground truth的窗口
        win_info = batch_win_info[batch_idx][j]  # one sample in window_info.log
        # the following four attributes are produced by the above one 
        # winInfo sample, 108 kinds of anchors are the
        # combination of different layer types and scale ratios
        # 预测结果
        result_class = batch_result_class[batch_idx][j]
        result_xmin = batch_result_xmin[batch_idx][j]
        result_xmax = batch_result_xmax[batch_idx][j]
        result_conf = batch_result_conf[batch_idx][j]

        num_box = len(result_class)  # (16*5+8*5+4*5) = sum of num_anchors*num_dbox

        video_name = win_info[1]
        tmp_df['video_name'] = [video_name] * num_box   
        tmp_df['start'] = [int(win_info[0])] * num_box
        tmp_df['conf'] = result_conf
        tmp_df['xmin'] = result_xmin
        tmp_df['xmax'] = result_xmax

        tmp_df.xmin = numpy.maximum(tmp_df.xmin, 0)
        tmp_df.xmax = numpy.minimum(tmp_df.xmax, config.window_size)
        # 获取真实的帧起始和结束，因为预测时都是从0开始的
        tmp_df.xmin = tmp_df.xmin + tmp_df.start
        tmp_df.xmax = tmp_df.xmax + tmp_df.start

        for cidx in range(config.num_classes):
            tmp_df['score_' + str(cidx)] = result_class[:, cidx]

        if not config.save_predict_result:
            # filter len(tmpDf) from 108 to ~20~40~
            tmp_df = tmp_df[tmp_df.score_0 < config.filter_neg_threshold]
        out_df = pandas.concat([out_df, tmp_df])

    return out_df  # 获取到了所有的行的80个bbox的结果


def final_result_process(stage, pretrain_dataset, config, mode, method, method_temporal='', df=None):
    if stage == 'fuse':
        if method_temporal == '':
            method_temporal = method
        spatial_file = join('results', 'predict_spatial_' + pretrain_dataset + '_' + method + '.csv')
        temporal_file = join('results', 'predict_temporal_' + pretrain_dataset + '_' + method_temporal + '.csv')
        if not os.path.isfile(spatial_file):
            print ("Error: spatial_file", spatial_file, "not exists!")
            exit()
        if not os.path.isfile(temporal_file):
            print ("Error: temporal_file", temporal_file, "not exists!")
            exit()
        df = fuse_two_stream(spatial_file, temporal_file)
        if method != method_temporal:
            method = method + '4sp_' + method_temporal + '4temp'
        result_file = join('results', 'result_fuse_' + pretrain_dataset + '_' + method + '.txt')
    else:
        result_file = join('results', 'result_' + mode + '_' + pretrain_dataset + '_' + method + '.txt')

    # necessary, otherwise the new content will append to the old
    if os.path.isfile(result_file):
        os.remove(result_file)
    df = df[df.score_0 < config.filter_neg_threshold]
    # it seems that without the following line,
    # the performance would be a little better
    df = df[df.conf > config.filter_conf_threshold]
    video_name_list = list(set(df.video_name.values[:]))
    # print "len(video_name_list):", len(video_name_list) # 210

    # 对每一个视频名的数据处理
    for video_name in video_name_list:
        tmpdf = df[df.video_name == video_name]
        tmpdf = post_process(tmpdf, config)

        # assign cliffDiving class as diving class too
        cliff_diving_df = tmpdf[tmpdf.out_type == 22]
        diving_df = cliff_diving_df
        diving_df.loc[:, 'out_type'] = 26
        tmpdf = pd.concat([tmpdf, diving_df])

        temporal_nms(config, tmpdf, result_file, video_name)   # 处理获得txt文件
        
