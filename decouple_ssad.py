# -*- coding: utf-8 -*-
"""
@author: HYPJUDY 2019/4/15
https://github.com/HYPJUDY

Decoupling Localization and Classification in Single Shot Temporal Action Detection
-----------------------------------------------------------------------------------

Train, test and post-processing for Decouple-SSAD

Usage:
Please refer to `run.sh` for details.
e.g.
`python decouple_ssad.py test UCF101 temporal decouple_ssad decouple_ssad`

"""

from operations import *
from load_data import get_train_data, get_test_data
from config import Config, get_models_dir, get_predict_result_path
import time
from os.path import join
import sys

####################################### PARAMETERS ########################################

stage = sys.argv[1]  # train/test/fuse/train_test_fuse
pretrain_dataset = sys.argv[2]  # UCF101/KnetV3
mode = sys.argv[3]  # temporal/spatial
method = sys.argv[4]
method_temporal = sys.argv[5]  # used for final result fusing

if (mode == 'spatial' and pretrain_dataset == 'Anet') or pretrain_dataset == 'KnetV3':
    feature_dim = 2048
else:
    feature_dim = 1024

models_dir = get_models_dir(mode, pretrain_dataset, method)
models_file_prefix = join(models_dir, 'model-ep')
test_checkpoint_file = join(models_dir, 'model-ep-30')
predict_file = get_predict_result_path(mode, pretrain_dataset, method)


######################################### TRAIN ##########################################

def train_operation(X, Y_label, Y_bbox, Index, LR, config):
    bsz = config.batch_size  
    ncls = config.num_classes 

    #######第一步 构建网络，包括基本特征提取网络，main_anchor_layer,提议和分类分支########
    net = base_feature_network(X)   # 这里也是得到一系列特征图，之后这些特征分层送入后面的mulClsReg_predict_layer等得到最终结果，再在这些最终特征图上做手脚
    MALs = main_anchor_layer(net)
    pBALs = branch_anchor_layer(MALs, 'ProposalBranch')
    cBALs = branch_anchor_layer(MALs, 'ClassificationBranch')

    ######第二步  定义一系列列表存储各个网络的结果########
    # --------------------------- Main Stream -----------------------------
    full_mainAnc_class = tf.reshape(tf.constant([]), [bsz, -1, ncls])
    full_mainAnc_conf = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmin = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmax = tf.reshape(tf.constant([]), [bsz, -1])

    full_mainAnc_BM_x = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_BM_w = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_BM_labels = tf.reshape(tf.constant([], dtype=tf.int32), [bsz, -1, ncls])
    full_mainAnc_BM_scores = tf.reshape(tf.constant([]), [bsz, -1])

    # ------------------ Localization/Proposal Branch -----------------------
    full_locAnc_conf = tf.reshape(tf.constant([]), [bsz, -1])
    full_locAnc_xmin = tf.reshape(tf.constant([]), [bsz, -1])
    full_locAnc_xmax = tf.reshape(tf.constant([]), [bsz, -1])

    full_locAnc_BM_x = tf.reshape(tf.constant([]), [bsz, -1])
    full_locAnc_BM_w = tf.reshape(tf.constant([]), [bsz, -1])
    full_locAnc_BM_scores = tf.reshape(tf.constant([]), [bsz, -1])

    # -------------------- Classification Branch ----------------------------
    full_clsAnc_class = tf.reshape(tf.constant([]), [bsz, -1, ncls])

    full_clsAnc_BM_labels = tf.reshape(tf.constant([], dtype=tf.int32), [bsz, -1, ncls])

    #######第三步  依次对各个层操作#######
    for i, ln in enumerate(config.layers_name): 
        mainAnc = mulClsReg_predict_layer(config, MALs[i], ln, 'mainStream')  # (1,80,24)
        locAnc = biClsReg_predict_layer(config, pBALs[i], ln, 'ProposalBranch') # (1,80,4)
        clsAnc = mulClsReg_predict_layer(config, cBALs[i], ln, 'ClassificationBranch') # (1,80,24)

        # adopt a simple average fusion strategy to fuse the location info of proposal branch
        # and main stream, and the class scores of classification branch and main stream.
        # Note that we also fuse the location info of classification branch and main stream.
        # Although the calculation of location is independent on classification,
        # the calculation of classification is partly depend on location.
        # Read the code of anchor_bboxes_encode and loss function for details.
        
        # cls_main:[32,80,22] 
        # loc_main:[32,80,2]
        cls_main, loc_main = tf.split(mainAnc, [ncls + 1, 2], axis=2)
        
        # others_propBranch：[32,80,2]
        # loc_propBranch:[32,80,2]
        others_propBranch, loc_propBranch = tf.split(locAnc, [1 + 1, 2], axis=2)
        
        # cls_clsBranch:[32,80,22]
        # loc_clsBranch:[32,80,2] 
        cls_clsBranch, loc_clsBranch = tf.split(clsAnc, [ncls + 1, 2], axis=2)
        
        clsAnc = tf.concat([(cls_main + cls_clsBranch) / 2, (loc_clsBranch + loc_main) / 2], axis=2)
        locAnc = tf.concat([others_propBranch, (loc_propBranch + loc_main) / 2], axis=2)

        # --------------------------- Main Stream -----------------------------
        
        """
          ##anchors_class, anchors_conf, anchors_rx, anchors_rw中保存的是所有的预测结果
          ##batch_match_x, batch_match_w, batch_match_labels, batch_match_scores中保存的所有的anchor和gt的匹配情况
        """
        [mainAnc_BM_x, mainAnc_BM_w, mainAnc_BM_labels, mainAnc_BM_scores,
         mainAnc_class, mainAnc_conf, mainAnc_rx, mainAnc_rw] = \
            anchor_bboxes_encode(mainAnc, Y_label, Y_bbox, Index, config, ln) 

        mainAnc_xmin = mainAnc_rx - mainAnc_rw / 2
        mainAnc_xmax = mainAnc_rx + mainAnc_rw / 2

        ######预测结果
        full_mainAnc_class = tf.concat([full_mainAnc_class, mainAnc_class], axis=1)
        full_mainAnc_conf = tf.concat([full_mainAnc_conf, mainAnc_conf], axis=1)
        full_mainAnc_xmin = tf.concat([full_mainAnc_xmin, mainAnc_xmin], axis=1)
        full_mainAnc_xmax = tf.concat([full_mainAnc_xmax, mainAnc_xmax], axis=1)

        #####和gt的匹配情况
        full_mainAnc_BM_x = tf.concat([full_mainAnc_BM_x, mainAnc_BM_x], axis=1)
        full_mainAnc_BM_w = tf.concat([full_mainAnc_BM_w, mainAnc_BM_w], axis=1)
        full_mainAnc_BM_labels = tf.concat([full_mainAnc_BM_labels, mainAnc_BM_labels], axis=1)
        full_mainAnc_BM_scores = tf.concat([full_mainAnc_BM_scores, mainAnc_BM_scores], axis=1)

        # ------------------ Localization/Proposal Branch -----------------------
        [locAnc_BM_x, locAnc_BM_w, _, locAnc_BM_scores,
         _, locAnc_conf, locAnc_rx, locAnc_rw] = \
            anchor_bboxes_encode(locAnc, Y_label, Y_bbox, Index, config, ln)

        locAnc_xmin = locAnc_rx - locAnc_rw / 2
        locAnc_xmax = locAnc_rx + locAnc_rw / 2

        full_locAnc_conf = tf.concat([full_locAnc_conf, locAnc_conf], axis=1)
        full_locAnc_xmin = tf.concat([full_locAnc_xmin, locAnc_xmin], axis=1)
        full_locAnc_xmax = tf.concat([full_locAnc_xmax, locAnc_xmax], axis=1)

        full_locAnc_BM_x = tf.concat([full_locAnc_BM_x, locAnc_BM_x], axis=1)
        full_locAnc_BM_w = tf.concat([full_locAnc_BM_w, locAnc_BM_w], axis=1)
        full_locAnc_BM_scores = tf.concat([full_locAnc_BM_scores, locAnc_BM_scores], axis=1)

        # -------------------- Classification Branch ----------------------------
        [_, _, clsAnc_BM_labels, _, clsAnc_class, _, _, _] = \
            anchor_bboxes_encode(clsAnc, Y_label, Y_bbox, Index, config, ln)

        full_clsAnc_class = tf.concat([full_clsAnc_class, clsAnc_class], axis=1)

        full_clsAnc_BM_labels = tf.concat([full_clsAnc_BM_labels, clsAnc_BM_labels], axis=1)

    main_class_loss, main_loc_loss, main_conf_loss = \
        loss_function(full_mainAnc_class, full_mainAnc_conf,
                      full_mainAnc_xmin, full_mainAnc_xmax,
                      full_mainAnc_BM_x, full_mainAnc_BM_w,
                      full_mainAnc_BM_labels, full_mainAnc_BM_scores, config)

    # Cls & Prop Branch loss
    cls_class_loss, loc_loc_loss, loc_conf_loss = \
        loss_function(full_clsAnc_class, full_locAnc_conf,
                      full_locAnc_xmin, full_locAnc_xmax,
                      full_locAnc_BM_x, full_locAnc_BM_w,
                      full_clsAnc_BM_labels, full_locAnc_BM_scores, config)

    class_loss = (main_class_loss + cls_class_loss * 2) / 3
    loc_loss = (main_loc_loss + loc_loc_loss * 2) / 3
    conf_loss = loc_conf_loss

    loss = class_loss + config.p_loc * loc_loss + config.p_conf * conf_loss

    trainable_variables = get_trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss, var_list=trainable_variables)

    return optimizer, loss, trainable_variables


def train_main(config):
    bsz = config.batch_size  

    tf.set_random_seed(config.seed)
    
    X = tf.placeholder(tf.float32, shape=(bsz, config.input_steps, feature_dim))
    Y_label = tf.placeholder(tf.int32, [None, config.num_classes])
    Y_bbox = tf.placeholder(tf.float32, [None, 3])  # 3 for 归一化的起始帧，归一化的结束帧，overlap_ratio
    Index = tf.placeholder(tf.int32, [bsz + 1])
    LR = tf.placeholder(tf.float32)

    # 这是最重要的函数，该函数获取数据后执行训练
    optimizer, loss, trainable_variables = \
        train_operation(X, Y_label, Y_bbox, Index, LR, config)

    model_saver = tf.train.Saver(var_list=trainable_variables, max_to_keep=2)

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

    tf.global_variables_initializer().run()

    # initialize parameters or restore from previous model
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if os.listdir(models_dir) == [] or config.initialize:
        init_epoch = 0
        print ("Initializing Network")
    else:
        init_epoch = int(config.steps)
        restore_checkpoint_file = join(models_dir, 'model-ep-' + str(config.steps - 1))
        model_saver.restore(sess, restore_checkpoint_file)

    #获取训练需要的数据
    #其中的每一个列表都存放了所有的数据，列表中的每一个元素都存放了一个batch的数据
    batch_train_dataX, batch_train_gt_label, batch_train_gt_info, batch_train_index = \
        get_train_data(config, mode, pretrain_dataset, True)
    
    num_batch_train = len(batch_train_dataX)  # 有多少个batch的数据要加载，定义时一个batch为32，则总的数据为4741，那么batch的数量为4741/32

    for epoch in range(init_epoch, config.training_epochs):

        loss_info = []
        # 每一次训练的是一个batch的数据，loss_info中记录的是每个batch的训练loss
        for idx in range(num_batch_train):
            feed_dict = {X: batch_train_dataX[idx],
                         Y_label: batch_train_gt_label[idx],
                         Y_bbox: batch_train_gt_info[idx],
                         Index: batch_train_index[idx],
                         LR: config.learning_rates[epoch]}
            _, out_loss = sess.run([optimizer, loss], feed_dict=feed_dict)

            loss_info.append(out_loss)
        # 最终的loss是每个batch的loss的均值
        print ("Training epoch ", epoch, " loss: ", np.mean(loss_info))

        if epoch == config.training_epochs - 2 or epoch == config.training_epochs - 1:
            model_saver.save(sess, models_file_prefix, global_step=epoch)


########################################### TEST ############################################

def test_operation(X, config):
    bsz = config.batch_size
    ncls = config.num_classes

    net = base_feature_network(X)
    MALs = main_anchor_layer(net)
    pBALs = branch_anchor_layer(MALs, 'ProposalBranch')
    cBALs = branch_anchor_layer(MALs, 'ClassificationBranch')

    full_clsAnc_class = tf.reshape(tf.constant([]), [bsz, -1, ncls])
    full_locAnc_conf = tf.reshape(tf.constant([]), [bsz, -1])
    full_locAnc_xmin = tf.reshape(tf.constant([]), [bsz, -1])
    full_locAnc_xmax = tf.reshape(tf.constant([]), [bsz, -1])

    for i, ln in enumerate(config.layers_name):
        mainAnc = mulClsReg_predict_layer(config, MALs[i], ln, 'mainStream')

        locAnc = biClsReg_predict_layer(config, pBALs[i], ln, 'ProposalBranch')

        clsAnc = mulClsReg_predict_layer(config, cBALs[i], ln, 'ClassificationBranch')

        cls_main, loc_main = tf.split(mainAnc, [ncls + 1, 2], axis=2)
        others_propBranch, loc_propBranch = tf.split(locAnc, [1 + 1, 2], axis=2)
        cls_clsBranch, loc_clsBranch = tf.split(clsAnc, [ncls + 1, 2], axis=2)
        
        # 将mainAnc拆分后融合到分类和定位分支中
        clsAnc = tf.concat([(cls_main + cls_clsBranch) / 2, (loc_clsBranch + loc_main) / 2], axis=2)
        locAnc = tf.concat([others_propBranch, (loc_propBranch + loc_main) / 2], axis=2)

        # 测试的时候只调用了box_adjust
        clsAnc_class, _, _, _ = anchor_box_adjust(clsAnc, config, ln)   # 分类分支获取分类得分
        _, locAnc_conf, locAnc_rx, locAnc_rw = anchor_box_adjust(locAnc, config, ln)  # 提议分支获取置信度，中心点和宽度

        locAnc_xmin = locAnc_rx - locAnc_rw / 2
        locAnc_xmax = locAnc_rx + locAnc_rw / 2

        full_clsAnc_class = tf.concat([full_clsAnc_class, clsAnc_class], axis=1)
        full_locAnc_conf = tf.concat([full_locAnc_conf, locAnc_conf], axis=1)
        full_locAnc_xmin = tf.concat([full_locAnc_xmin, locAnc_xmin], axis=1)
        full_locAnc_xmax = tf.concat([full_locAnc_xmax, locAnc_xmax], axis=1)
        
    # 测试是的分类概率是经过softmax后得到的
    full_clsAnc_class = tf.nn.softmax(full_clsAnc_class, dim=-1)  
    return full_clsAnc_class, full_locAnc_conf, full_locAnc_xmin, full_locAnc_xmax


def test_main(config):
    batch_dataX, batch_winInfo = get_test_data(config, mode, pretrain_dataset)

    X = tf.placeholder(tf.float32, shape=(config.batch_size, config.input_steps, feature_dim))

    anchors_class, anchors_conf, anchors_xmin, anchors_xmax = test_operation(X, config)

    model_saver = tf.train.Saver()
    '''
    tf.InteractiveSession()来构建会话的时候，我们可以先构建一个session然后再定义操作（operation），
    如果我们使用tf.Session()来构建会话我们需要在会话构建之前定义好全部的操作（operation）然后再构建会话。
    
    '''
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    tf.global_variables_initializer().run()
    model_saver.restore(sess, test_checkpoint_file)

    batch_result_class = []
    batch_result_conf = []
    batch_result_xmin = []
    batch_result_xmax = []

    num_batch = len(batch_dataX)
    for idx in range(num_batch):
        out_anchors_class, out_anchors_conf, out_anchors_xmin, out_anchors_xmax = \
            sess.run([anchors_class, anchors_conf, anchors_xmin, anchors_xmax],
                     feed_dict={X: batch_dataX[idx]})
        batch_result_class.append(out_anchors_class)
        batch_result_conf.append(out_anchors_conf)
        batch_result_xmin.append(out_anchors_xmin * config.window_size)
        batch_result_xmax.append(out_anchors_xmax * config.window_size)

    outDf = pd.DataFrame(columns=config.outdf_columns)

    # 将结果写入csv文件
    for i in range(num_batch):
        tmpDf = result_process(batch_winInfo, batch_result_class, batch_result_conf,
                               batch_result_xmin, batch_result_xmax, config, i)

        outDf = pd.concat([outDf, tmpDf])
    if config.save_predict_result:
        outDf.to_csv(predict_file, index=False)
    return outDf


if __name__ == "__main__":
    config = Config()
    start_time = time.time()
    elapsed_time = 0
    if stage == 'train':
        train_main(config)
        elapsed_time = time.time() - start_time
    elif stage == 'test':
        df = test_main(config)
        elapsed_time = time.time() - start_time
        final_result_process(stage, pretrain_dataset, config, mode, method, '', df)
    elif stage == 'fuse':
        final_result_process(stage, pretrain_dataset, config, mode, method, method_temporal)
        elapsed_time = time.time() - start_time
    elif stage == 'train_test_fuse':
        train_main(config)
        elapsed_time = time.time() - start_time
        tf.reset_default_graph()
        df = test_main(config)
        final_result_process(stage, pretrain_dataset, config, mode, method, '', df)
    else:
        print ("No stage", stage, "Please choose a stage from train/test/fuse/train_test_fuse.")
    print ("Elapsed time:", elapsed_time, "start time:", start_time)
    
    
