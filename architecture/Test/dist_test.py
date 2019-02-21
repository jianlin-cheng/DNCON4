# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:33:30 2018

@author: Jie Hou
"""
import sys
import os
sys.path.insert(0, '/mnt/data/zhiye/Python/DNCON4/architecture/lib/')

from Model_construct import *
from DNCON_lib import *
import numpy as np
import math
import random

dist_filepath = '/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/dist/'
dist_mat_filepath = '/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/dist_mat/'
seq_len_file = '/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/lists-test-train/L.txt'
log_file = '/mnt/data/zhiye/Python/DNCON4/architecture/Test/dist_test.log'

def dist2distmat(dist_filepath, dist_mat_filepath, seq_len_file, log_file):
    f = open(seq_len_file, 'r')
    fl = open(log_file, 'w')
    for line in f.readlines():
        single_line = line.strip()
        single_line = single_line.split(' ')
        if len(single_line) != 2:
            print('Wrong file!')
        seq_name = single_line[0]
        seq_len = int(single_line[1])
        print('process ', seq_name, '\n')
        dist_array = np.full((seq_len, seq_len), -1, dtype = np.float32)
        np.fill_diagonal(dist_array, 0) #i=j means self to self, dist=0
        dist_filename = dist_filepath + seq_name + '.dist'
        if not os.path.exists(dist_filename):
            print('%s dist file not exit continue\n'%seq_name)
        df = open(dist_filename, 'r')
        wrong_len_flag = 0
        for df_line in df.readlines():
            df_single_line = df_line.strip()
            df_single_line = df_single_line.split(' ')
            if len(df_single_line) != 3:
                print('wrong line! continue')
                continue
            X = int(df_single_line[0])-1
            Y = int(df_single_line[1])-1
            D = float(df_single_line[2])
            if X >= seq_len or Y >= seq_len:
                wrong_len_flag = 1
                break
            dist_array[X, Y] = D
            dist_array[Y, X] = D
        if wrong_len_flag == 1:
            print('worng len ',seq_name)
            fl.write('worng len %s\n' % seq_name)
        dist_mat_filename = dist_mat_filepath + seq_name + '.distm'
        # with open(dist_mat_filename, "a") as myfile: myfile.write(dist_array)
        np.savetxt(dist_mat_filename, dist_array, fmt='%.4f')

import linecache
import  math
chn_filepath = '/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/chains_wrong/'
dist_filepath = '/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/dist_wrong/'


def chains2dist(chn_filepath, dist_filepath):
    files = os.listdir(chn_filepath)
    for file in files:
        if not os.path.isdir(file):
            chn_filename = chn_filepath + '/' +file
            fn = open(chn_filename, 'r')
            seq_len = len(open(chn_filename, 'r').readlines())
            seq_name = str(file).split('.')[0]
            dist_filename = dist_filepath + seq_name+'.dist'
            print('process ', seq_name)
            for i in range(1, seq_len+1):
                single_line_i = linecache.getline(chn_filename, i).strip().split(' ')
                # single_line = single_line.strip()
                # single_line = single_line.split(' ')
                single_line_i = [x for x in single_line_i if x != '']
                Xi = float(single_line_i[6])
                Yi = float(single_line_i[7])
                Zi = float(single_line_i[8])
                for j in range(i+1, seq_len+1):
                    single_line_j = linecache.getline(chn_filename, j).strip().split(' ')
                    single_line_j = [x for x in single_line_j if x != '']
                    Xj = float(single_line_j[6])
                    Yj = float(single_line_j[7])
                    Zj = float(single_line_j[8])

                    dist = math.sqrt(pow((Xi-Xj), 2)+pow((Yi-Yj), 2)+pow((Zi-Zj), 2))

                    dist_output = "%i %i %.4f\n" % (i, j, dist)
                    with open(dist_filename, "a") as myfile: myfile.write(dist_output)

cmap_filepath='/mnt/data/zhiye/Python/DNCON4/architecture/outputs/CNN_arch/sample120/filter64_layers2_optnadam_ftsize5_batchsize1_glorot_uniform_binary_crossentropy_relu_18.0_0.6737/predict_casp13_map/'

# cmap_filepath='/mnt/data/zhiye/Python/DNCON4/architecture/outputs/CNN_arch/COV3455/filter64_layers2_optnadam_ftsize5_batchsize2_glorot_uniform_binary_crossentropy_relu_5.0/predict_casp13_map/'predict_map
path_of_lists = '/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/lists-test-train/'
path_of_X='/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/feats/'
path_of_Y='/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/feats/'
reject_fea_file =  '/mnt/data/zhiye/Python/DNCON4/architecture/lib/feature_to_use_lewis.txt'
def evaluate_from_cmap(cmap_filepath, path_of_lists, path_of_X, path_of_Y, reject_fea_file):
    Maximum_length=300
    tr_l, tr_n, tr_e, te_l, te_n, te_e = build_dataset_dictionaries(path_of_lists)
    all_l = te_l.copy()
    all_n = te_n.copy()
    all_e = te_e.copy()
    all_l.update(tr_l)
    all_n.update(tr_n)
    all_e.update(tr_e)
    # print('Total Number of Training and Test dataset = ',str(len(all_l)))
    sys.stdout.flush()
    # print('Load all test data into memory..', end='')
    selected_list = subset_pdb_dict(te_l, 0, Maximum_length, Maximum_length, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
    # print('Loading data sets ..', end='')

    testdata_len_range = 50
    step_num = 0
    out_avg_pc_l5 = 0.0
    out_avg_pc_l2 = 0.0
    out_avg_pc_1l = 0.0
    out_avg_acc_l5 = 0.0
    out_avg_acc_l2 = 0.0
    out_avg_acc_1l = 0.0
    for key in selected_list:
        # print('saving cmap of %s\n' % (key))
        value = selected_list[key]
        p1 = {key: value}
    # for i in range(0, 300, testdata_len_range):
    #     p1 = {key: value for key, value in selected_list.items() if value < i + testdata_len_range and value >= i}
        p2 = list(p1.keys())
        p3 = list(p1.values())
        Maximum_length = value
        # print('this batchsize = ', len(p1))
        if len(p1) < 1:
            continue
        # print("start predict")
        # selected_list_2D = get_x_2D_from_this_list(p1, path_of_X, Maximum_length, '80', reject_fea_file, p3)
        # print("\nselected_list_2D.shape: ", selected_list_2D.shape)
        # print('Loading label sets..')
        selected_list_label = get_y_from_this_list(p1, path_of_Y, 5, Maximum_length, '80')
        # feature_2D_num = selected_list_2D.shape[3]
        # DNCON4_CNN.load_weights(model_weight_out)
        # DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_2D], batch_size=1)
        DNCON4_CNN_prediction = np.zeros((len(p2),Maximum_length*Maximum_length))
        for p2_index in range(0, len(p2)):
            seq_name = p2[p2_index]
            L = p3[p2_index]
            print('this seq_name = ', seq_name)
            print('this L = ', L)
            cmap_filename = cmap_filepath + seq_name + '.txt'
            s_cmap = np.fromfile(cmap_filename)
            Y = np.zeros((Maximum_length, Maximum_length))
            i=0
            with open(cmap_filename) as f:
                for line in f:
                    this_line = line.strip().split()
                    Y[i, 0:L] = np.asarray(this_line)
                    i = i + 1
            for p in range(0, L):
                for q in range(0, L):
                    # updated only for the last project 'p19' to test the effect
                    if (abs(q - p) < 24):
                        Y[p][q] = 0
            Y = Y.flatten()
            DNCON4_CNN_prediction[p2_index, :] = Y

        DNCON4_CNN_prediction = DNCON4_CNN_prediction.reshape(len(p1), Maximum_length * Maximum_length)
        (list_acc_l5, list_acc_l2, list_acc_1l, avg_pc_l5, avg_pc_l2, avg_pc_1l, avg_acc_l5, avg_acc_l2,
         avg_acc_1l) = evaluate_prediction(p1, all_n, all_e, DNCON4_CNN_prediction, selected_list_label, 5)
        out_avg_pc_l5 += avg_pc_l5 * len(p1)
        out_avg_pc_l2 += avg_pc_l2 * len(p1)
        out_avg_pc_1l += avg_pc_1l * len(p1)
        out_avg_acc_l5 += avg_acc_l5 * len(p1)
        out_avg_acc_l2 += avg_acc_l2 * len(p1)
        out_avg_acc_1l += avg_acc_1l * len(p1)

        step_num += 1
    print('step_num=', step_num)
    all_num = len(selected_list)
    out_avg_pc_l5 /= all_num
    out_avg_pc_l2 /= all_num
    out_avg_pc_1l /= all_num
    out_avg_acc_l5 /= all_num
    out_avg_acc_l2 /= all_num
    out_avg_acc_1l /= all_num
    val_acc_history_content = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (out_avg_pc_l5, out_avg_pc_l2, out_avg_pc_1l, out_avg_acc_l5, out_avg_acc_l2,
    out_avg_acc_1l)
    print(val_acc_history_content)


crop_file = '/mnt/data/zhiye/Python/DNCON4/architecture/Test/X-119L-A-ccmpred-60.crop'
import struct
def xshow(filename, nx, nz):
    f = open(filename, "rb")
    pic = np.zeros((nx, nz))
    for i in range(nx):
        for j in range(nz):
            data = f.read(4)
            elem = struct.unpack("f", data)[0]
            pic[i][j] = elem
    f.close()
    return pic


weights=[2, 1]
weights = K.variable(weights)

def cacluate_loss(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss

evaluate_from_cmap(cmap_filepath, path_of_lists, path_of_X, path_of_Y, reject_fea_file)