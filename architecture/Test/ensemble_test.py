import sys
import os
sys.path.insert(0, '/mnt/data/zhiye/Python/DNCON4/architecture/lib/')

from Model_construct import *
from DNCON_lib import *

from Model_construct import _weighted_binary_crossentropy, _weighted_categorical_crossentropy

import numpy as np
import time
import shutil
import platform
import pickle
from six.moves import range

from keras.models import model_from_json,load_model, Sequential, Model
from keras.utils import CustomObjectScope
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adagrad, Adamax, Nadam
from random import randint


# CV_dir = "/mnt/data/zhiye/Python/DNCON4/architecture/outputs/CNN_arch/COV3455/filter64_layers2_optnadam_ftsize5_batchsize2_glorot_uniform_binary_crossentropy_relu_5.0/"
# CV_dir = "/mnt/data/zhiye/Python/DNCON4/architecture/outputs/CNN_arch/new_fea_test_gen/filter64_layers2_optnadam_ftsize5_batchsize1_glorot_uniform_binary_crossentropy_relu_1.0/"
CV_dir = "/mnt/data/zhiye/Python/DNCON4/architecture/outputs/CNN_arch/sample120/filter64_layers2_optnadam_ftsize5_batchsize1_glorot_uniform_binary_crossentropy_relu_18.0_0.6737/"
feature_dir = '/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/feats/'
model_prefix = 'DNCON4_2dCONV'
filtsize = str(5)
win_array = list(map(int,filtsize.split("_")))
use_bias = True
hidden_type = 'sigmoid'
nb_filters = 64
nb_layers = 2
opt = 'nadam'
path_of_lists = '/mnt/data/zhiye/Python/DNCON4/data/CASP13/lists-test-train/'
path_of_X='/mnt/data/zhiye/Python/DNCON4/data/CASP13/feats/'
path_of_Y='/mnt/data/zhiye/Python/DNCON4/data/CASP13/feats/'
# path_of_lists = '/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/lists-test-train/'
# path_of_X='/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/feats/'
# path_of_Y='/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/feats/'
Maximum_length = 700
dist_string = '80'
reject_fea_file =  '/mnt/data/zhiye/Python/DNCON4/architecture/lib/feature_to_use_lewis.txt'
initializer="glorot_uniform"
loss_function="binary_crossentropy"
weight_p=18

def DNCON4_1d2dconv_train_win_filter_layer_opt_fast_2D_generator(CV_dir, feature_dir,
                                                                 model_prefix,win_array, use_bias, hidden_type, nb_filters,
                                                                 nb_layers, opt, path_of_lists, path_of_Y,
                                                                 path_of_X, Maximum_length, dist_string,
                                                                 reject_fea_file='None',
                                                                 initializer="he_normal",
                                                                 loss_function="weighted_crossentropy", weight_p=1.0,
                                                                 weight_n=1.0, list_sep_flag=False, activation="relu"):

    print("\n######################################\n佛祖保佑，永不迨机，永无bug，精度九十九\n######################################\n")

    train_avg_acc_l5_best = 0
    val_avg_acc_l5_best = 0
    feature_2D_num = 441
    ### Define the model
    model_out = "%s/model-train-%s.json" % (CV_dir, model_prefix)
    model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir, model_prefix)
    model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir, model_prefix)

    if model_prefix == 'DNCON4_2dCONV':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
        DNCON4_CNN = DeepConv_with_paras_2D(win_array, feature_2D_num, use_bias, hidden_type, nb_filters, nb_layers,
                                            opt, initializer, loss_function, weight_p, weight_n, activation)
    elif model_prefix == 'DNCON4_2dINCEP':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        DNCON4_CNN = DeepInception_with_paras_2D(win_array, feature_2D_num, use_bias, hidden_type, nb_filters,
                                                 nb_layers, opt, initializer, loss_function, weight_p, weight_n)
    elif model_prefix == 'DNCON4_2dRES':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
        DNCON4_CNN = DeepResnet_with_paras_2D(win_array, feature_2D_num, use_bias, hidden_type, nb_filters, nb_layers,
                                              opt, initializer, loss_function, weight_p, weight_n)
    elif model_prefix == 'DNCON4_2dRCNN':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#0.001
        DNCON4_CNN = DeepCovRCNN_with_paras_2D(win_array, feature_2D_num, use_bias, hidden_type, nb_filters, nb_layers,
                                               opt, initializer, loss_function, weight_p, weight_n)
    else:
        DNCON4_CNN = DeepConv_with_paras_2D(win_array, feature_2D_num, use_bias, hidden_type, nb_filters, nb_layers,
                                            opt)

    best_val_acc_out = "%s/best_validation3.acc_history" % (CV_dir)

    chkdirs(best_val_acc_out)
    with open(best_val_acc_out, "a") as myfile:
        myfile.write("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\n")

    if loss_function == 'weighted_crossentropy':
        loss_function = _weighted_binary_crossentropy(weight_p, weight_n)
    else:
        loss_function = loss_function
    DNCON4_CNN.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)

    # with CustomObjectScope({
    #     'MaxoutConv2D_Test': MaxoutConv2D_Test,
    #     }):
    #     json_string = open(model_out).read()
    #     DNCON4_CNN = model_from_json(json_string)


    DNCON4_CNN.load_weights(model_weight_out_best)
    # model_weight_epochs = "%s/model_weights/" % (CV_dir)
    # model_predict = "%s/predict_map/" % (CV_dir)
    model_test_casp13 = "%s/predict_casp13_map/" % (CV_dir)
    # chkdirs(model_weight_epochs)
    # chkdirs(model_predict)
    chkdirs(model_test_casp13)

    te_l = build_dataset_dictionaries_test(path_of_lists)
    selected_list = subset_pdb_dict(te_l,  0, 700, 5000, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
    print('Loading data sets ..',end='')

    testdata_len_range = 50
    step_num = 0
    out_avg_pc_l5 = 0.0
    out_avg_pc_l2 = 0.0
    out_avg_pc_1l = 0.0
    out_avg_acc_l5 = 0.0
    out_avg_acc_l2 = 0.0
    out_avg_acc_1l = 0.0
    for key in selected_list:
        value = selected_list[key]
        p1 = {key: value}
        Maximum_length = value
        print(key)
        if len(p1) < 1:
            continue
        print("start predict")
        selected_list_2D = get_x_2D_from_this_list(p1, path_of_X, Maximum_length, dist_string, reject_fea_file, value)
        print("selected_list_2D.shape: ", selected_list_2D.shape)
        # print('Loading label sets..')
        # selected_list_label = get_y_from_this_list(p1, path_of_Y, 24, Maximum_length, dist_string)
        feature_2D_num = selected_list_2D.shape[3]
        
        
        DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_2D], batch_size=1)
        CMAP = DNCON4_CNN_prediction.reshape(Maximum_length, Maximum_length)
        Map_UpTrans = np.triu(CMAP, 1).T
        Map_UandL = np.triu(CMAP)
        real_cmap = Map_UandL + Map_UpTrans
        # weights = os.listdir(model_weight_epochs)
        # max_cmap = np.zeros((Maximum_length, Maximum_length))
        # weight_num = 0
        # for weight in weights:
        #     model_weight_out = model_weight_epochs + '/' + weight

        #     DNCON4_CNN.load_weights(model_weight_out)
        #     DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_2D], batch_size=1)

        #     CMAP = DNCON4_CNN_prediction.reshape(Maximum_length, Maximum_length)
        #     Map_UpTrans = np.triu(CMAP, 1).T
        #     Map_UandL = np.triu(CMAP)
        #     real_cmap = Map_UandL + Map_UpTrans
            
        #     # max_cmap += real_cmap
        #     # weight_num += 1
        #     max_cmap = [[max_cmap[i][j] if max_cmap[i][j] < real_cmap[i][j] else real_cmap[i][j] for j in range(Maximum_length)] for i in range(Maximum_length)]
        # real_cmap = max_cmap

        cmap_file = "%s/%s.txt" % (model_test_casp13, key)
        np.savetxt(cmap_file, real_cmap, fmt='%.4f')

        # DNCON4_CNN_prediction = np.array(real_cmap).reshape(len(p1), Maximum_length * Maximum_length)
        # (list_acc_l5, list_acc_l2, list_acc_1l, avg_pc_l5, avg_pc_l2, avg_pc_1l, avg_acc_l5, avg_acc_l2, avg_acc_1l) = evaluate_prediction_4(p1, DNCON4_CNN_prediction, selected_list_label, 24)
    #     val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key, value, avg_pc_l5, avg_pc_l2, avg_pc_1l, avg_acc_l5, avg_acc_l2, avg_acc_1l)
    #     print('The best validation accuracy is ', val_acc_history_content)
    #     with open(best_val_acc_out, "a") as myfile:
    #         myfile.write(val_acc_history_content)
    #     out_avg_pc_l5 += avg_pc_l5 * len(p1)
    #     out_avg_pc_l2 += avg_pc_l2 * len(p1)
    #     out_avg_pc_1l += avg_pc_1l * len(p1)
    #     out_avg_acc_l5 += avg_acc_l5 * len(p1)
    #     out_avg_acc_l2 += avg_acc_l2 * len(p1)
    #     out_avg_acc_1l += avg_acc_1l * len(p1)

        step_num += 1
    print('step_num=', step_num)
    # all_num = len(selected_list)
    # out_avg_pc_l5 /= all_num
    # out_avg_pc_l2 /= all_num
    # out_avg_pc_1l /= all_num
    # out_avg_acc_l5 /= all_num
    # out_avg_acc_l2 /= all_num
    # out_avg_acc_1l /= all_num
    # val_acc_history_content = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (out_avg_pc_l5, out_avg_pc_l2, out_avg_pc_1l, out_avg_acc_l5, out_avg_acc_l2, out_avg_acc_1l)

    # print('The validation accuracy is ', val_acc_history_content)

    # return out_avg_acc_l5
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DNCON4_1d2dconv_train_win_filter_layer_opt_fast_2D_generator(CV_dir, feature_dir,
                                                                 model_prefix,win_array, use_bias, hidden_type, nb_filters,
                                                                 nb_layers, opt, path_of_lists, path_of_Y,
                                                                 path_of_X, Maximum_length, dist_string,reject_fea_file,
                                                                 initializer, loss_function, weight_p,list_sep_flag=False, activation="relu")