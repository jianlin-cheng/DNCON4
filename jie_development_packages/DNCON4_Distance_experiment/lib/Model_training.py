# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:47:26 2017

@author: Jie Hou
"""
import os

from Model_construct import *
from DNCON_lib import *

from Model_construct import _weighted_binary_crossentropy, _weighted_categorical_crossentropy, _weighted_mean_squared_error

from maxout_test import MaxoutConv2D_Test

import numpy as np
import time
import shutil
import shlex, subprocess
from subprocess import Popen, PIPE
import sys
import os
from shutil import copyfile
import platform
import gc
from collections import defaultdict
import pickle
from six.moves import range

import keras.backend as K
import tensorflow as tf
from keras.models import model_from_json,load_model, Sequential, Model
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adagrad, Adamax, Nadam
from keras.utils import multi_gpu_model
from keras.utils.generic_utils import Progbar
from keras.constraints import maxnorm
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Convolution1D, Convolution2D
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from random import randint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

# dist_string = '80'interval
def generate_data_from_file(path_of_lists, path_of_X, path_of_Y, min_seq_sep,dist_string, batch_size, reject_fea_file='None', 
    child_list_index=0, list_sep_flag=False, dataset_select='train', if_use_binsize=False, predict_method='bin_class'):
    import pickle
    accept_list = []
    if reject_fea_file != 'None':
        with open(reject_fea_file) as f:
            for line in f:
                if line.startswith('#'):
                    feature_name = line.strip()
                    feature_name = feature_name[0:]
                    accept_list.append(feature_name)
    if (dataset_select == 'train'):
        dataset_list = build_dataset_dictionaries_train(path_of_lists)
    elif (dataset_select == 'vali'):
        dataset_list = build_dataset_dictionaries_test(path_of_lists)
    else:
        dataset_list = build_dataset_dictionaries_train(path_of_lists)

    if (list_sep_flag == False):
        training_dict = subset_pdb_dict(dataset_list, 0, 700, 5000, 'random') #can be random ordered   
        training_list = list(training_dict.keys())
        training_lens = list(training_dict.values())
        all_data_num = len(training_dict)
        loopcount = all_data_num // int(batch_size)
    else:
        training_dict = subset_pdb_dict(dataset_list, 0, 700, 5000, 'ordered') #can be random ordered
        all_training_list = list(training_dict.keys())
        all_training_lens = list(training_dict.values())
        if ((child_list_index + 1) * 15 > len(training_dict)):
            print("Out of list range!\n")
            child_list_index = len(training_dict)/15 - 1
        child_batch_list = all_training_list[child_list_index * 15:(child_list_index + 1) * 15]
        child_batch_list_len = all_training_lens[child_list_index * 15:(child_list_index + 1) * 15]
        all_data_num = 15
        loopcount = all_data_num // int(batch_size)
        print('crop_list_num=',all_data_num)
        print('crop_loopcount=',loopcount)
        training_list = child_batch_list
        training_lens = child_batch_list_len
    while(True):
        index = randint(0, loopcount-1)
        # print('\nindex=', index)
        batch_list = training_list[index * batch_size:(index + 1) * batch_size]
        batch_list_len = training_lens[index * batch_size:(index + 1) * batch_size]
        if if_use_binsize:
            max_pdb_lens = 320
        else:
            max_pdb_lens = max(batch_list_len)

        data_all_dict = dict()
        batch_X=[]
        batch_Y=[]
        for i in range(0, len(batch_list)):
            pdb_name = batch_list[i]
            pdb_len = batch_list_len[i]
            notxt_flag = True
            #featurefile = path_of_X + '/X-' + pdb_name + '.txt'
            featurefile = '/storage/htc/bdm/DNCON4/feature/other/output/'+pdb_name+'/feat-'+pdb_name+'.txt'
            if ((len(accept_list) == 1 and ('# cov' not in accept_list and '# plm' not in accept_list)) or 
                  (len(accept_list) == 2 and ('# cov' not in accept_list or '# plm' not in accept_list and '# dist error' not in accept_list)) or (len(accept_list) > 2)):
                notxt_flag = False
                if not os.path.isfile(featurefile):
                    print("feature file not exists: ",featurefile, " pass!")
                    continue     
            cov = path_of_X + '/' + pdb_name + '.cov'
            if '# cov' in accept_list:
                if not os.path.isfile(cov):
                    print("Cov Matrix file not exists: ",cov, " pass!")
                    continue        
            plm = '/storage/htc/bdm/DNCON4/feature/plm/cull_plm/output/' + pdb_name + '.plm'
            if '# plm' in accept_list:
                if not os.path.isfile(plm):
                    print("plm matrix file not exists: ",plm, " pass!")
                    continue
            #distpred = path_of_X + '/dist_pred/' + pdb_name + '-distance.txt'
            distpred = '/storage/htc/bdm/DNCON4/data/cullpdb_dataset/distance_prediction20190315/pred_distance/' + pdb_name + '-distance.txt'
            if '# dist error' in accept_list:      
              if not os.path.isfile(distpred):
                          print("dist error file not exists: ",distpred, " pass!")
                          continue

            if predict_method == 'bin_class':       
                targetfile = path_of_Y + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'mul_class':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue 
            elif predict_method == 'real_dist':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'real_dist_limited':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'real_dist_limited2':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'real_dist_limited3':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'real_dist_limited4':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'real_dist_limited16':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'real_dist_limited17':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'real_dist_limited19':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'real_dist_limited21':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'real_dist_limited22':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'real_dist_limited23':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'real_dist_limited25':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'real_dist_scaled':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif predict_method == 'dist_error':
                targetfile1 = path_of_Y + pdb_name + '.txt'
                targetfile2 = '/storage/htc/bdm/DNCON4/data/cullpdb_dataset/distance_prediction20190315/Error_distance/' + pdb_name + '-DistError.txt'
                if not os.path.isfile(targetfile1) or not os.path.isfile(targetfile2):
                        print("target file not exists: ",targetfile1, targetfile2, " pass!")
                        continue
            else:
                targetfile = path_of_Y + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue

            #(featuredata, feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, accept_list, pdb_len, notxt_flag)
            (featuredata, feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, distpred, accept_list, pdb_len, notxt_flag)
            feature_2D_all = []
            for key in sorted(feature_index_all_dict.keys()):
                featurename = feature_index_all_dict[key]
                feature = featuredata[key]
                feature = np.asarray(feature)
                if feature.shape[0] == feature.shape[1]:
                    feature_2D_all.append(feature)
                else:
                    print("Wrong dimension")
            fea_len = feature_2D_all[0].shape[0]

            F = len(feature_2D_all)
            X = np.zeros((max_pdb_lens, max_pdb_lens, F))
            for m in range(0, F):
                X[0:fea_len, 0:fea_len, m] = feature_2D_all[m]

            # X = np.memmap(cov, dtype=np.float32, mode='r', shape=(F, max_pdb_lens, max_pdb_lens))
            # X = X.transpose(1, 2, 0)

            l_max = max_pdb_lens
            if predict_method == 'bin_class':
                Y = getY(targetfile, min_seq_sep, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
            elif predict_method == 'mul_class':
                Y1 = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y1)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                #Y1 = np.random.randint(0,5,(4,4))
                Y1 = Y1.reshape(l_max, l_max, 1) #contains class id
                max_class=42
                Y= (np.arange(max_class) == Y1[...,None]).astype(int) # L*L*1*42
                Y = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[3]) # 1*L*L*42
                #print(pdb_name,": ",Y.shape)
                #print("Haven't has this function! quit!\n")
                #sys.exit(1)
            elif predict_method == 'real_dist':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
            elif predict_method == 'real_dist_limited':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>20] = 20
            elif predict_method == 'real_dist_limited2':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>15] = 15
            elif predict_method == 'real_dist_limited3':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>18] = 18
            elif predict_method == 'real_dist_limited4':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>24] = 24
            elif predict_method == 'real_dist_limited16':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>16] = 16
            elif predict_method == 'real_dist_limited17':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>17] = 17
            elif predict_method == 'real_dist_limited19':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>19] = 19
            elif predict_method == 'real_dist_limited21':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>21] = 21
            elif predict_method == 'real_dist_limited22':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>2] = 22
            elif predict_method == 'real_dist_limited23':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>23] = 23
            elif predict_method == 'real_dist_limited25':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>25] = 25
            elif predict_method == 'dist_error':
                Y1 = getY(targetfile1, 0, l_max) #real dist
                Y2 = getY(targetfile2, 0, l_max) #dist error
                if (l_max * l_max != len(Y1) or l_max * l_max != len(Y2)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y1=Y1.reshape(l_max, l_max, 1)
                Y2=Y2.reshape(l_max, l_max, 1)
                Y1[Y1>20] = 20
                
                #print("Mean of Y1: ",np.mean(Y1))
                #print("Mean of Y2: ",np.mean(Y2))
                Y = np.concatenate((Y1,Y2), axis=-1)
            elif predict_method == 'real_dist_scaled':
                Y1 = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y1)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y1 = Y1.reshape(l_max, l_max, 1)
                #Y = 1/(1 + (Y1/3.0)*(Y1/3.0)) #plot 1-2/(e^(0.1*x)+1), -10<x<100
                Y = 1-2/(np.exp(0.1*Y1)+1)
                Y[Y>1]=1
                Y[Y<0]=0
                #Y = 1/(1 + K.square(Y/3.0))
                # real_dist is different with bin class, bin out is l*l vector, real dist out is (l,l) matrix
                # Y = getY_dist(targetfile, 0, l_max)
                # if (l_max != len(Y)):
                #     print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                #     print('len(Y) = %d, lmax = %d'%(len(Y), l_max))
                #     continue
                
            
            batch_X.append(X)
            batch_Y.append(Y)
        batch_X =  np.array(batch_X)
        batch_Y =  np.array(batch_Y)
        # print('X shape', batch_X.shape)
        # print('Y shape', batch_Y.shape)
        if len(batch_X.shape) < 4 or len(batch_Y.shape) < 4:
            # print('Data shape error, pass!\n')
            continue
        yield batch_X, batch_Y

def DNCON4_1d2dconv_train_win_filter_layer_opt_fast_2D_generator(feature_num,CV_dir,feature_dir,model_prefix,
    epoch_outside,epoch_inside,epoch_rerun,interval_len,seq_end,win_array,use_bias,hidden_type,nb_filters,nb_layers,opt,
    lib_dir, batch_size_train,path_of_lists, path_of_Y, path_of_X, Maximum_length,dist_string, reject_fea_file='None',
    initializer = "he_normal", loss_function = "weighted_BCE", weight_p=1.0, weight_n=1.0,  list_sep_flag=False,  if_use_binsize = False,dataset='dncon2'): 


    start=0
    end=seq_end
    import numpy as np
    Train_data_keys = dict()
    Train_targets_keys = dict()
    print("\n######################################\n佛祖保佑，永不迨机，永无bug，精度九十九\n######################################\n")
    feature_2D_num=feature_num # the number of features for each residue
 
    print("Load feature number", feature_2D_num)
    ### Define the model 
    model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
    model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
    model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)
    model_and_weights = "%s/model-weight-%s.h5" % (CV_dir,model_prefix)

        # opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)#1
        # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
        # opt = SGD(lr=0.001, momentum=0.9, decay=0.00, nesterov=False)
        # opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06, decay=0.0)
        # opt = Adagrad(lr=0.01, epsilon=1e-06)
        # opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    if model_prefix == 'DNCON4_2dCONV':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#0.00
        DNCON4_CNN = DeepConv_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dRES':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
        print(feature_2D_num)
        DNCON4_CNN = DeepResnet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    else:
        DNCON4_CNN = DeepConv_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt)

    rerun_flag=0
    # with tf.device("/cpu:0"):
    #     DNCON4_CNN = multi_gpu_model(DNCON4_CNN, gpus=2)
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    best_val_acc_out = "%s/best_validation.acc_history" % (CV_dir)
    if os.path.exists(model_weight_out_best):
        # print("######## Loading existing weights ",model_weight_out)
        # DNCON4_CNN.load_weights(model_weight_out)
        print("######## Loading existing weights ",model_weight_out_best)
        DNCON4_CNN.load_weights(model_weight_out_best)
        rerun_flag = 1
    else:
        print("######## Setting initial weights")   
    
        chkdirs(train_acc_history_out)     
        with open(train_acc_history_out, "a") as myfile:
          myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n")
          
        chkdirs(val_acc_history_out)     
        with open(val_acc_history_out, "a") as myfile:
          myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n")
        
        chkdirs(best_val_acc_out)     
        with open(best_val_acc_out, "a") as myfile:
          myfile.write("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n")

    #predict_method has three value : bin_class, mul_class, real_dist
    predict_method = 'bin_class'
    if dataset == 'dncon2':
      if loss_function == 'weighted_BCE':
          predict_method = 'bin_class'
          path_of_Y_train = path_of_Y + '/bin_class/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          if weight_p <= 1:
              weight_n = 1.0 - weight_p
          loss_function = _weighted_binary_crossentropy(weight_p, weight_n)
      elif loss_function == 'unweighted_BCE':
          predict_method = 'bin_class'
          path_of_Y_train = path_of_Y + '/bin_class/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = 'binary_crossentropy'
      elif loss_function == 'weighted_CCE':
          predict_method = 'mul_class'
          loss_function = _weighted_categorical_crossentropy(weight_p)
      elif loss_function == 'MSE_limited':
          predict_method = 'real_dist_limited'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = 'mean_squared_error'
      elif loss_function == 'MSE_limited2':
          predict_method = 'real_dist_limited2'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = 'mean_squared_error'
      elif loss_function == 'MSE_limited3':
          predict_method = 'real_dist_limited3'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = 'mean_squared_error'
      elif loss_function == 'weighted_MSE':
          predict_method = 'real_dist'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited':
          predict_method = 'real_dist_limited'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited2':
          predict_method = 'real_dist_limited2'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited3':
          predict_method = 'real_dist_limited3'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited4':
          predict_method = 'real_dist_limited4'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited16':
          predict_method = 'real_dist_limited16'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited17':
          predict_method = 'real_dist_limited17'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited19':
          predict_method = 'real_dist_limited19'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'sigmoid_MSE':
          predict_method = 'real_dist_scaled'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = 'mean_squared_error'
      elif loss_function == 'categorical_crossentropy':
          predict_method = 'mul_class'
          path_of_Y_train = path_of_Y + '/dist_map/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = 'categorical_crossentropy'
      elif loss_function == 'weighted_MSElimited20_disterror':
          predict_method = 'dist_error'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = _weighted_mean_squared_error(1)
      else:
          predict_method = 'real_dist'
          path_of_Y_train = path_of_Y + '/real_dist/'
          path_of_Y_evalu = path_of_Y + '/bin_class/'
          loss_function = loss_function
    elif dataset=='cullpdb':
      if loss_function == 'weighted_BCE':
          predict_method = 'bin_class'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          if weight_p <= 1:
              weight_n = 1.0 - weight_p
          loss_function = _weighted_binary_crossentropy(weight_p, weight_n)
      elif loss_function == 'unweighted_BCE':
          predict_method = 'bin_class'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = 'binary_crossentropy'
      elif loss_function == 'weighted_CCE':
          predict_method = 'mul_class'
          loss_function = _weighted_categorical_crossentropy(weight_p)
      elif loss_function == 'MSE_limited':
          predict_method = 'real_dist_limited'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = 'mean_squared_error'
      elif loss_function == 'MSE_limited2':
          predict_method = 'real_dist_limited2'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = 'mean_squared_error'
      elif loss_function == 'MSE_limited3':
          predict_method = 'real_dist_limited3'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = 'mean_squared_error'
      elif loss_function == 'weighted_MSE':
          predict_method = 'real_dist'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited':
          predict_method = 'real_dist_limited'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited2':
          predict_method = 'real_dist_limited2'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited3':
          predict_method = 'real_dist_limited3'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited4':
          predict_method = 'real_dist_limited4'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited16':
          predict_method = 'real_dist_limited16'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited17':
          predict_method = 'real_dist_limited17'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited19':
          predict_method = 'real_dist_limited19'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited21':
          predict_method = 'real_dist_limited21'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited22':
          predict_method = 'real_dist_limited22'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited23':
          predict_method = 'real_dist_limited23'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'weighted_MSE_limited25':
          predict_method = 'real_dist_limited25'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      elif loss_function == 'sigmoid_MSE':
          predict_method = 'real_dist_scaled'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = 'mean_squared_error'
      elif loss_function == 'categorical_crossentropy':
          predict_method = 'mul_class'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = 'categorical_crossentropy'
      elif loss_function == 'weighted_MSElimited20_disterror':
          predict_method = 'dist_error'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = _weighted_mean_squared_error(1)
      else:
          predict_method = 'real_dist'
          path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
          path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
          loss_function = loss_function
    else:
        print('The dataset is wrong: ',dataset)
        exit(-1)
    
    print("Setting predict_method to ",predict_method)
    print("Setting loss function to ",loss_function)

    DNCON4_CNN.compile(loss=loss_function, metrics=['acc'], optimizer=opt)

    model_weight_epochs = "%s/model_weights/"%(CV_dir)
    model_predict= "%s/predict_map/"%(CV_dir)
    model_predict_casp13= "%s/predict_map_casp13/"%(CV_dir)
    model_val_acc= "%s/val_acc_inepoch/"%(CV_dir)
    chkdirs(model_weight_epochs)
    chkdirs(model_predict)
    chkdirs(model_predict_casp13)
    chkdirs(model_val_acc)

    tr_l = build_dataset_dictionaries_train(path_of_lists)
    te_l = build_dataset_dictionaries_test(path_of_lists)
    all_l = te_l.copy()
    train_data_num = len(tr_l)
    child_list_num = int(train_data_num/15)# 15 is the inter
    print('Total Number of Training dataset = ',str(len(tr_l)))

    # callbacks=[reduce_lr]
    train_avg_acc_l5_best = 0 
    val_avg_acc_l5_best = 0
    min_seq_sep = 0
    lr_decay = False
    train_loss_last = 1e32
    train_loss_list = []
    evalu_loss_list = []
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    for epoch in range(epoch_rerun,epoch_outside):
        if (epoch >=30 and lr_decay == False):
            print("Setting lr_decay as true")
            lr_decay = True
            opt = SGD(lr=0.001, momentum=0.9, decay=0.00, nesterov=False)
            DNCON4_CNN.load_weights(model_weight_out_best)
            DNCON4_CNN.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)
            #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=5, min_lr=0.00005)


        # class_weight = {0:1.,1:60.}
        if(list_sep_flag == False):
            print("\n############ Running epoch ", epoch)
            if epoch == 0 and rerun_flag == 0:
                first_inepoch = 1
                history = DNCON4_CNN.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, '80', batch_size_train, reject_fea_file, if_use_binsize=if_use_binsize, predict_method=predict_method), steps_per_epoch = len(tr_l)//batch_size_train, epochs = first_inepoch, 
                    validation_data = generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, '80', batch_size_train, reject_fea_file, dataset_select='vali', if_use_binsize=if_use_binsize, predict_method=predict_method), validation_steps = len(te_l))           
                train_loss_list.append(history.history['loss'][first_inepoch-1])
                evalu_loss_list.append(history.history['val_loss'][first_inepoch-1])
            else:
                history = DNCON4_CNN.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, '80', batch_size_train, reject_fea_file, if_use_binsize=if_use_binsize, predict_method=predict_method), steps_per_epoch = len(tr_l)//batch_size_train, epochs = 1,
                    validation_data = generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, '80', batch_size_train, reject_fea_file, dataset_select='vali', if_use_binsize=if_use_binsize, predict_method=predict_method), validation_steps = len(te_l))  
                train_loss_list.append(history.history['loss'][0])
                evalu_loss_list.append(history.history['val_loss'][0])
        else:
            for index in range(child_list_num):
                print("\n############ Runing outside epoch %i\nProcessing list %i of %i...."%(epoch, index, child_list_num))
                history = DNCON4_CNN.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, '80', batch_size_train, reject_fea_file, index, list_sep_flag=True), steps_per_epoch = 15//batch_size_train, epochs = 5)

        DNCON4_CNN.save_weights(model_weight_out)

        # DNCON4_CNN.save(model_and_weights)
        
        ### save models
        model_weight_out_inepoch = "%s/model-train-weight-%s-epoch%i.h5" % (model_weight_epochs,model_prefix,epoch)
        DNCON4_CNN.save_weights(model_weight_out_inepoch)
        ##### running validation

        print("Now evaluate for epoch ",epoch)
        val_acc_out_inepoch = "%s/validation_epoch%i.acc_history" % (model_val_acc, epoch) 
        sys.stdout.flush()
        print('Load all test data into memory..',end='')
        selected_list = subset_pdb_dict(te_l,   0, 500, 5000, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
        print('Loading data sets ..',end='')

        testdata_len_range=50
        step_num = 0
        out_avg_pc_l5 = 0.0
        out_avg_pc_l2 = 0.0
        out_avg_pc_1l = 0.0
        out_avg_acc_l5 = 0.0
        out_avg_acc_l2 = 0.0
        out_avg_acc_1l = 0.0
        out_gloable_mse = 0.0
        out_weighted_mse = 0.0
        out_precision_all_long = 0.0
        out_recall_all_long = 0.0
        out_fscore_all_long = 0.0
        out_gloable_error_mse=0.0
        
        for key in selected_list:
            value = selected_list[key]
            p1 = {key: value}
            if if_use_binsize:
                Maximum_length = 320
            else:
                Maximum_length = value
            print(len(p1))
            if len(p1) < 1:
                continue
            print("start predict")
            selected_list_2D = get_x_2D_from_this_list(p1, path_of_X, Maximum_length,dist_string, reject_fea_file, value)

            #print("selected_list_2D.shape: ",selected_list_2D.shape)
            print('Loading label sets..')
            selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, Maximum_length, dist_string)# dist_string 80
            feature_2D_num = selected_list_2D.shape[3]
            DNCON4_CNN.load_weights(model_weight_out)

            DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_2D], batch_size= 1)
            
            if predict_method == 'mul_class':
                ### convert back to <8 probability 
                #DNCON4_CNN_prediction.shape
                DNCON4_CNN_prediction= DNCON4_CNN_prediction[:,:,:,0:8].sum(axis=-1)
                #DNCON4_CNN_prediction.shape
            
            if predict_method == 'dist_error':
                ### convert back to <8 probability 
                #DNCON4_CNN_prediction.shape
                #print("DNCON4_CNN_prediction.shape: ",DNCON4_CNN_prediction.shape)
                DNCON4_CNN_prediction_disterror = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction_dist = DNCON4_CNN_prediction[:,:,:,0]
                DNCON4_CNN_prediction_error = DNCON4_CNN_prediction[:,:,:,1]
                print("Mean of DNCON4_CNN_prediction_dist: ",np.mean(DNCON4_CNN_prediction_dist))
                print("Mean of DNCON4_CNN_prediction_error: ",np.mean(DNCON4_CNN_prediction_error))
                #print("DNCON4_CNN_prediction_dist.shape: ",DNCON4_CNN_prediction_dist.shape)
                #print("DNCON4_CNN_prediction_error.shape: ",DNCON4_CNN_prediction_error.shape)
                DNCON4_CNN_prediction= np.copy(DNCON4_CNN_prediction_dist)
                #print("Mean of DNCON4_CNN_prediction_dist2: ",np.mean(DNCON4_CNN_prediction_dist))
                #print("Mean of DNCON4_CNN_prediction_error2: ",np.mean(DNCON4_CNN_prediction_error))
                ## get error
                CMAP_error = DNCON4_CNN_prediction_error.reshape(Maximum_length, Maximum_length)
                Map_UpTrans = np.triu(CMAP_error, 1).T
                Map_UandL = np.triu(CMAP_error)
                real_cmap_error = Map_UandL + Map_UpTrans
                DNCON4_CNN_prediction_error = real_cmap_error.reshape(len(p1), Maximum_length*Maximum_length)
        
            CMAP = DNCON4_CNN_prediction.reshape(Maximum_length, Maximum_length)
            Map_UpTrans = np.triu(CMAP, 1).T
            Map_UandL = np.triu(CMAP)
            real_cmap = Map_UandL + Map_UpTrans

            DNCON4_CNN_prediction = real_cmap.reshape(len(p1), Maximum_length*Maximum_length)
            
            global_mse = 0.0
            weighted_mse = 0.0
            error_global_mse = 0.0
            if predict_method == 'real_dist':
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
            elif predict_method == 'real_dist_scaled':
                ### convert back to distance 
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0]=0.001
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>=1]=0.999
                #DNCON4_CNN_prediction = 3*np.sqrt((1-DNCON4_CNN_prediction)/DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 10*np.log((1+DNCON4_CNN_prediction)/(1-DNCON4_CNN_prediction))
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                # to binary
                #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
            elif predict_method == 'real_dist_limited':
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                # to binary
                #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
            elif predict_method == 'real_dist_limited2' or predict_method == 'real_dist_limited3' or predict_method == 'real_dist_limited4':
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                # to binary
                #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
            elif predict_method == 'real_dist_limited16' or predict_method == 'real_dist_limited17' or predict_method == 'real_dist_limited18' or predict_method == 'real_dist_limited19':
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                # to binary
                #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
            elif predict_method == 'real_dist_limited20' or predict_method == 'real_dist_limited21' or predict_method == 'real_dist_limited22' or predict_method == 'real_dist_limited23':
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                # to binary
                #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
            elif predict_method == 'real_dist_limited24' or predict_method == 'real_dist_limited25':
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                # to binary
                #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
            elif predict_method == 'dist_error':
                selected_list_label_disterror = get_y_from_this_list_dist_error(p1, path_of_Y_train, 0, Maximum_length)# dist_string 80
                global_mse, weighted_mse,error_global_mse = evaluate_prediction_dist_error_4(DNCON4_CNN_prediction_disterror, selected_list_label_disterror)
                # to binary
                #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
                
            

            (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = evaluate_prediction_4(p1, DNCON4_CNN_prediction, selected_list_label, 24)
            
            #### calculate  recall and precision for all long range
            #pred_contact = floor_lower_left_to_zero(DNCON4_CNN_prediction, 24)
            #datacount = len(selected_list_label[:, 0])
            #true_contact = floor_lower_left_to_zero(selected_list_label, 24)            
            #pred_contact_flatten = pred_contact.flatten()          
            #true_contact_flatten = true_contact.flatten()
            #precision_all_long = precision_score(true_contact_flatten,pred_contact_flatten)
            #recall_all_long = recall_score(true_contact_flatten,pred_contact_flatten)
            #fscore_all_long = f1_score(true_contact_flatten,pred_contact_flatten)
            
            #val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l,global_mse,weighted_mse,precision_all_long,recall_all_long,fscore_all_long)
            if predict_method == 'dist_error':
              val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l,global_mse,weighted_mse,error_global_mse)
              #print("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\tPrecision_all_long\tRecall_all_long\tFscore_all_long\n")
              print("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\tGlobal_Error_MSE\n")            
            else:
              val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l,global_mse,weighted_mse)
              #print("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\tPrecision_all_long\tRecall_all_long\tFscore_all_long\n")
              print("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n")
            print('The best validation accuracy is ',val_acc_history_content)
            with open(val_acc_out_inepoch, "a") as myfile:
                myfile.write(val_acc_history_content)
            DNCON4_CNN_prediction = DNCON4_CNN_prediction.reshape (Maximum_length, Maximum_length)
            # cmap_file = "%s/%s.txt" % (model_predict,key)
            # np.savetxt(cmap_file, DNCON4_CNN_prediction, fmt='%.4f')
            cmap_file = "%s/%s.txt" % (model_predict,key)
            np.savetxt(cmap_file, real_cmap, fmt='%.4f')
            
            if predict_method == 'dist_error':
              DNCON4_CNN_prediction_error = DNCON4_CNN_prediction_error.reshape (Maximum_length, Maximum_length)
              # cmap_file = "%s/%s.txt" % (model_predict,key)
              # np.savetxt(cmap_file, DNCON4_CNN_prediction, fmt='%.4f')
              cmap_file = "%s/%s-DistError.txt" % (model_predict,key)
              np.savetxt(cmap_file, real_cmap_error, fmt='%.4f')
              out_gloable_error_mse += error_global_mse
            

            
            out_gloable_mse += global_mse
            out_weighted_mse += weighted_mse 
            out_avg_pc_l5 += avg_pc_l5 * len(p1)
            out_avg_pc_l2 += avg_pc_l2 * len(p1)
            out_avg_pc_1l += avg_pc_1l * len(p1)
            out_avg_acc_l5 += avg_acc_l5 * len(p1)
            out_avg_acc_l2 += avg_acc_l2 * len(p1)
            out_avg_acc_1l += avg_acc_1l * len(p1)
            #out_precision_all_long += precision_all_long
            #out_recall_all_long += recall_all_long
            #out_fscore_all_long += fscore_all_long
            
            step_num += 1
        print ('step_num=', step_num)
        all_num = len(selected_list)
        out_gloable_error_mse /= all_num
        out_gloable_mse /= all_num
        out_weighted_mse /= all_num
        out_avg_pc_l5 /= all_num
        out_avg_pc_l2 /= all_num
        out_avg_pc_1l /= all_num
        out_avg_acc_l5 /= all_num
        out_avg_acc_l2 /= all_num
        out_avg_acc_1l /= all_num
        #out_precision_all_long /= all_num
        #out_recall_all_long /= all_num
        #out_fscore_all_long /= all_num
        
        if predict_method == 'dist_error':
          val_acc_history_content = "%i\t%i\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (interval_len,epoch,epoch_inside,out_avg_pc_l5,out_avg_pc_l2,out_avg_pc_1l,
              out_avg_acc_l5,out_avg_acc_l2,out_avg_acc_1l, out_gloable_mse, out_weighted_mse,out_gloable_error_mse)
        else:
          val_acc_history_content = "%i\t%i\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (interval_len,epoch,epoch_inside,out_avg_pc_l5,out_avg_pc_l2,out_avg_pc_1l,
              out_avg_acc_l5,out_avg_acc_l2,out_avg_acc_1l, out_gloable_mse, out_weighted_mse)
        with open(val_acc_history_out, "a") as myfile:
                    myfile.write(val_acc_history_content)  

        train_loss = history.history['loss'][0]
        # train_loss = history.history['val_loss'][0]
        print("Train loss of epoch %i is %.6f" % (epoch, train_loss))
        if (lr_decay and train_loss_last != 1e32):
            current_lr = K.get_value(DNCON4_CNN.optimizer.lr)
            if (train_loss < train_loss_last and current_lr < 0.01):
                K.set_value(DNCON4_CNN.optimizer.lr, current_lr * 1.2)
                print("Increasing learning rate to {} ...".format(current_lr * 1.2))
            else:
                K.set_value(DNCON4_CNN.optimizer.lr, current_lr * 0.8)
                print("Decreasing learning rate to {} ...".format(current_lr * 0.8))
        train_loss_last = train_loss


        print('The validation accuracy is ',val_acc_history_content)
        if out_avg_acc_l5 >= val_avg_acc_l5_best:
            val_avg_acc_l5_best = out_avg_acc_l5 
            score_imed = "Accuracy L5 of Val: %.4f\t\n" % (val_avg_acc_l5_best)
            model_json = DNCON4_CNN.to_json()
            print("Saved model to disk")
            with open(model_out, "w") as json_file:
                json_file.write(model_json)
            print("Saved best weight to disk, ", score_imed)
            DNCON4_CNN.save_weights(model_weight_out_best)

        if epoch == epoch_outside-1:
            for key in selected_list:
                print('saving cmap of %s\n'%(key))
                value = selected_list[key]
                single_dict={key:value}
                if if_use_binsize:
                    Maximum_length = 320
                else:
                    Maximum_length = value
                # print(single_dict)
                selected_list_2D = get_x_2D_from_this_list(single_dict, path_of_X, Maximum_length, dist_string, reject_fea_file, value)
                print("selected_list_2D.shape: ",selected_list_2D.shape)
                print('Loading label sets..')
                selected_list_label = get_y_from_this_list(single_dict, path_of_Y, min_seq_sep, Maximum_length, dist_string)
                DNCON4_CNN.load_weights(model_weight_out_best)

                DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_2D], batch_size= 1)
                if predict_method == 'mul_class':
                    ### convert back to <8 probability 
                    #DNCON4_CNN_prediction.shape
                    DNCON4_CNN_prediction= DNCON4_CNN_prediction[:,:,:,0:8].sum(axis=-1)
                    #DNCON4_CNN_prediction.shape
                               
                if predict_method == 'dist_error':
                    ### convert back to <8 probability 
                    #DNCON4_CNN_prediction.shape
                    print("DNCON4_CNN_prediction.shape: ",DNCON4_CNN_prediction.shape)
                    DNCON4_CNN_prediction_disterror = np.copy(DNCON4_CNN_prediction)
                    DNCON4_CNN_prediction_dist = DNCON4_CNN_prediction[:,:,:,0]
                    DNCON4_CNN_prediction_error = DNCON4_CNN_prediction[:,:,:,1]
                    print("Mean of DNCON4_CNN_prediction_dist: ",np.mean(DNCON4_CNN_prediction_dist))
                    print("Mean of DNCON4_CNN_prediction_error: ",np.mean(DNCON4_CNN_prediction_error))
                    print("DNCON4_CNN_prediction_dist.shape: ",DNCON4_CNN_prediction_dist.shape)
                    print("DNCON4_CNN_prediction_error.shape: ",DNCON4_CNN_prediction_error.shape)
                    DNCON4_CNN_prediction= DNCON4_CNN_prediction_dist
                    print("Mean of DNCON4_CNN_prediction_dist2: ",np.mean(DNCON4_CNN_prediction_dist))
                    print("Mean of DNCON4_CNN_prediction_error2: ",np.mean(DNCON4_CNN_prediction_error))
                    ## get error
                    CMAP_error = DNCON4_CNN_prediction_error.reshape(Maximum_length, Maximum_length)
                    Map_UpTrans = np.triu(CMAP_error, 1).T
                    Map_UandL = np.triu(CMAP_error)
                    real_cmap_error = Map_UandL + Map_UpTrans
                    DNCON4_CNN_prediction_error = real_cmap_error.reshape(len(p1), Maximum_length*Maximum_length)
                    
                    CMAP = DNCON4_CNN_prediction.reshape(Maximum_length, Maximum_length)
                    Map_UpTrans = np.triu(CMAP, 1).T
                    Map_UandL = np.triu(CMAP)
                    real_cmap = Map_UandL + Map_UpTrans

                DNCON4_CNN_prediction = DNCON4_CNN_prediction.reshape(len(single_dict), Maximum_length*Maximum_length)
                
                if predict_method == 'real_dist':
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                    DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                    DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
                elif predict_method == 'real_dist_scaled':
                    ### convert back to distance 
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0]=0.001
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction>=1]=0.999
                    #DNCON4_CNN_prediction = 3*np.sqrt((1-DNCON4_CNN_prediction)/DNCON4_CNN_prediction)
                    DNCON4_CNN_prediction = 10*np.log((1+DNCON4_CNN_prediction)/(1-DNCON4_CNN_prediction))
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                    selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                    global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                    # to binary
                    #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                    DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                    DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
                elif predict_method == 'real_dist_limited':
                    selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                    global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                    # to binary
                    #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                    DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                    DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
                elif predict_method == 'real_dist_limited2' or predict_method == 'real_dist_limited3' or predict_method == 'real_dist_limited4':
                    selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                    global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                    # to binary
                    #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                    DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                    DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
                elif predict_method == 'real_dist_limited16' or predict_method == 'real_dist_limited17' or predict_method == 'real_dist_limited18' or predict_method == 'real_dist_limited19':
                    selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                    global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                    # to binary
                    #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                    DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                    DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
                elif predict_method == 'real_dist_limited20' or predict_method == 'real_dist_limited21' or predict_method == 'real_dist_limited22' or predict_method == 'real_dist_limited23':
                    selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                    global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                    # to binary
                    #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                    DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                    DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
                elif predict_method == 'real_dist_limited24' or predict_method == 'real_dist_limited25':
                    selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                    global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                    # to binary
                    #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                    DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                    DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
                elif predict_method == 'dist_error':
                    selected_list_label_disterror = get_y_from_this_list_dist_error(p1, path_of_Y_train, 0, Maximum_length)# dist_string 80
                    global_mse, weighted_mse,error_global_mse = evaluate_prediction_dist_error_4(DNCON4_CNN_prediction_disterror, selected_list_label_dist)
                    # to binary
                    #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                    #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                    DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                    DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                    DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
                
                (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = evaluate_prediction_4(single_dict, DNCON4_CNN_prediction, selected_list_label, 24)
                
                
                #### calculate  recall and precision for all long range
                #pred_contact = floor_lower_left_to_zero(DNCON4_CNN_pred, 24)
                #datacount = len(selected_list_label[:, 0])
                #true_contact = floor_lower_left_to_zero(selected_list_label, 24)            
                #pred_contact_flatten = pred_contact.flatten()          
                #true_contact_flatten = true_contact.flatten()
                #precision_all_long = precision_score(true_contact_flatten,pred_contact_flatten)
                #recall_all_long = recall_score(true_contact_flatten,pred_contact_flatten)
                #fscore_all_long = f1_score(true_contact_flatten,pred_contact_flatten)
                
                if predict_method == 'dist_error':
                  val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l,global_mse,weighted_mse,error_global_mse)
                  print("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\tGlobal_Error_MSE\n")            
                else:
                  val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l,global_mse,weighted_mse)
                  print("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n")
                print('The best validation accuracy is ',val_acc_history_content)
                with open(best_val_acc_out, "a") as myfile:
                    myfile.write(val_acc_history_content)  
                DNCON4_CNN_prediction = DNCON4_CNN_prediction.reshape (Maximum_length, Maximum_length)
                # cmap_file = "%s/%s.txt" % (model_predict,key)
                # np.savetxt(cmap_file, DNCON4_CNN_prediction, fmt='%.4f')
                cmap_file = "%s/%s.txt" % (model_predict,key)
                np.savetxt(cmap_file, real_cmap, fmt='%.4f')
                
                DNCON4_CNN_prediction_error = DNCON4_CNN_prediction_error.reshape (Maximum_length, Maximum_length)
                # cmap_file = "%s/%s.txt" % (model_predict,key)
                # np.savetxt(cmap_file, DNCON4_CNN_prediction, fmt='%.4f')
                cmap_file = "%s/%s-DistError.txt" % (model_predict,key)
                np.savetxt(cmap_file, real_cmap_error, fmt='%.4f')
                # history_loss_file = CV_dir+"/train_loss.history"
            
            ###Predict CASP13
            # path_of_lists = '/mnt/data/zhiye/Python/DNCON4/data/CASP13/lists-test-train/'
            # path_of_X='/mnt/data/zhiye/Python/DNCON4/data/CASP13/feats/'
            # te_l = build_dataset_dictionaries_test(path_of_lists)
            # selected_list = subset_pdb_dict(te_l,   0, 700, 5000, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
            # print('Loading casp13 data sets ..',end='')
            
            # for key in selected_list:
            #     value = selected_list[key]
            #     single_dict={key:value}
            #     Maximum_length = value
            #     print('saving cmap of', single_dict)
            #     # print(single_dict)
            #     selected_list_2D = get_x_2D_from_this_list(single_dict, path_of_X, Maximum_length, dist_string, reject_fea_file, value)
            #     print("selected_list_2D.shape: ",selected_list_2D.shape)
            #     DNCON4_CNN.load_weights(model_weight_out_best)

            #     DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_2D], batch_size= 1)

            #     CMAP = DNCON4_CNN_prediction.reshape(Maximum_length, Maximum_length)
            #     Map_UpTrans = np.triu(CMAP, 1).T
            #     Map_UandL = np.triu(CMAP)
            #     real_cmap = Map_UandL + Map_UpTrans

            #     cmap_file = "%s/%s.txt" % (model_predict_casp13,key)
            #     np.savetxt(cmap_file, real_cmap, fmt='%.4f')


        print("Train loss history:", train_loss_list)
        print("Validation loss history:", evalu_loss_list)
        #clear memory
        # K.clear_session()
        # tf.reset_default_graph()

    print("Training finished, best validation acc = ",val_avg_acc_l5_best)
    return val_avg_acc_l5_best




def DNCON4_1d2dconv_train_win_filter_layer_opt_fast_2D_predict(feature_num,CV_dir,feature_dir,model_prefix,
    epoch_outside,epoch_inside,epoch_rerun,interval_len,seq_end,win_array,use_bias,hidden_type,nb_filters,nb_layers,opt,
    lib_dir, batch_size_train,path_of_lists, path_of_Y, path_of_X, Maximum_length,dist_string, reject_fea_file='None',
    initializer = "he_normal", loss_function = "weighted_BCE", weight_p=1.0, weight_n=1.0,  list_sep_flag=False,  if_use_binsize = False): 


    start=0
    end=seq_end
    import numpy as np
    Train_data_keys = dict()
    Train_targets_keys = dict()
    feature_2D_num=feature_num # the number of features for each residue
 
    print("Load feature number", feature_2D_num)
    ### Define the model 
    model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
    model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
    #model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)
    model_weight_out_best = "%s/model-train-weight-%s-best-val-final.h5" % (CV_dir,model_prefix)
    model_and_weights = "%s/model-weight-%s.h5" % (CV_dir,model_prefix)

    if model_prefix == 'DNCON4_2dCONV':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#0.00
        DNCON4_CNN = DeepConv_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dRES':
        opt = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
        DNCON4_CNN = DeepResnet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    else:
        DNCON4_CNN = DeepConv_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt)

    rerun_flag=0
    # with tf.device("/cpu:0"):
    #     DNCON4_CNN = multi_gpu_model(DNCON4_CNN, gpus=2)
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    best_val_acc_out = "%s/best_validation.acc_history" % (CV_dir)
    if os.path.exists(model_weight_out_best):
        # print("######## Loading existing weights ",model_weight_out)
        # DNCON4_CNN.load_weights(model_weight_out)
        print("######## Loading existing weights ",model_weight_out_best)
        DNCON4_CNN.load_weights(model_weight_out_best)
        rerun_flag = 1
    else:
        print("######## Setting initial weights")   
    
    #predict_method has three value : bin_class, mul_class, real_dist
    predict_method = 'bin_class'
    if loss_function == 'weighted_BCE':
        predict_method = 'bin_class'
        path_of_Y_train = path_of_Y + '/bin_class/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        if weight_p <= 1:
            weight_n = 1.0 - weight_p
        loss_function = _weighted_binary_crossentropy(weight_p, weight_n)
    elif loss_function == 'unweighted_BCE':
        predict_method = 'bin_class'
        path_of_Y_train = path_of_Y + '/bin_class/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'binary_crossentropy'
    elif loss_function == 'weighted_CCE':
        predict_method = 'mul_class'
        loss_function = _weighted_categorical_crossentropy(weight_p)
    elif loss_function == 'MSE_limited':
        predict_method = 'real_dist_limited'
        path_of_Y_train = path_of_Y + '/real_dist/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'mean_squared_error'
    elif loss_function == 'MSE_limited2':
        predict_method = 'real_dist_limited2'
        path_of_Y_train = path_of_Y + '/real_dist/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'mean_squared_error'
    elif loss_function == 'MSE_limited3':
        predict_method = 'real_dist_limited3'
        path_of_Y_train = path_of_Y + '/real_dist/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'mean_squared_error'
    elif loss_function == 'weighted_MSE':
        predict_method = 'real_dist'
        path_of_Y_train = path_of_Y + '/real_dist/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = _weighted_mean_squared_error(1)
    elif loss_function == 'weighted_MSE_limited':
        predict_method = 'real_dist_limited'
        path_of_Y_train = path_of_Y + '/real_dist/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = _weighted_mean_squared_error(1)
    elif loss_function == 'sigmoid_MSE':
        predict_method = 'real_dist_scaled'
        path_of_Y_train = path_of_Y + '/real_dist/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'mean_squared_error'
    elif loss_function == 'categorical_crossentropy':
        predict_method = 'mul_class'
        path_of_Y_train = path_of_Y + '/dist_map/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'categorical_crossentropy'
    else:
        predict_method = 'real_dist'
        path_of_Y_train = path_of_Y + '/real_dist/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = loss_function
    print("Setting predict_method to ",predict_method)
    print("Setting loss function to ",loss_function)

    DNCON4_CNN.compile(loss=loss_function, metrics=['acc'], optimizer=opt)

    model_weight_epochs = "%s/model_weights/"%(CV_dir)
    model_predict= "%s/predict_map/"%(CV_dir)
    model_predict_casp13= "%s/predict_map_casp13/"%(CV_dir)
    model_val_acc= "%s/val_acc_inepoch/"%(CV_dir)
    model_predict_best= "%s/predict_map_best/"%(CV_dir)
    chkdirs(model_weight_epochs)
    chkdirs(model_predict)
    chkdirs(model_predict_casp13)
    chkdirs(model_val_acc)
    chkdirs(model_predict_best)

    tr_l = build_dataset_dictionaries_train(path_of_lists)
    te_l = build_dataset_dictionaries_test(path_of_lists)
    all_l = te_l.copy()
    train_data_num = len(tr_l)
    child_list_num = int(train_data_num/15)# 15 is the inter
    print('Total Number of Training dataset = ',str(len(tr_l)))

    # callbacks=[reduce_lr]
    train_avg_acc_l5_best = 0 
    val_avg_acc_l5_best = 0
    min_seq_sep = 0
    lr_decay = False
    train_loss_last = 1e32
    train_loss_list = []
    evalu_loss_list = []
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    for epoch in range(0,1):
        if (epoch >=30 and lr_decay == False):
            print("Setting lr_decay as true")
            lr_decay = True
            opt = SGD(lr=0.001, momentum=0.9, decay=0.00, nesterov=False)
            DNCON4_CNN.load_weights(model_weight_out_best)
            DNCON4_CNN.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=5, min_lr=0.00005)

        print("Now evaluate for epoch ",epoch)
        val_acc_out_inepoch = "%s/validation_epoch%i.acc_history" % (model_val_acc, epoch) 
        sys.stdout.flush()
        print('Load all test data into memory..',end='')
        selected_list = subset_pdb_dict(te_l,   0, 500, 5000, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
        print('Loading data sets ..',end='')

        testdata_len_range=50
        step_num = 0
        out_avg_pc_l5 = 0.0
        out_avg_pc_l2 = 0.0
        out_avg_pc_1l = 0.0
        out_avg_acc_l5 = 0.0
        out_avg_acc_l2 = 0.0
        out_avg_acc_1l = 0.0
        out_gloable_mse = 0.0
        out_weighted_mse = 0.0
        out_precision_all_long = 0.0
        out_recall_all_long = 0.0
        out_fscore_all_long = 0.0
        
        for key in selected_list:
            value = selected_list[key]
            p1 = {key: value}
            if if_use_binsize:
                Maximum_length = 320
            else:
                Maximum_length = value
            print(len(p1))
            if len(p1) < 1:
                continue
            print("start predict")
            selected_list_2D = get_x_2D_from_this_list(p1, path_of_X, Maximum_length,dist_string, reject_fea_file, value)

            print("selected_list_2D.shape: ",selected_list_2D.shape)
            print('Loading label sets..')
            selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, Maximum_length, dist_string)# dist_string 80
            feature_2D_num = selected_list_2D.shape[3]
            DNCON4_CNN.load_weights(model_weight_out)

            DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_2D], batch_size= 1)
            
            if predict_method == 'mul_class':
                ### convert back to <8 probability 
                #DNCON4_CNN_prediction.shape
                DNCON4_CNN_prediction= DNCON4_CNN_prediction[:,:,:,0:8].sum(axis=-1)
                #DNCON4_CNN_prediction.shape

            CMAP = DNCON4_CNN_prediction.reshape(Maximum_length, Maximum_length)
            
            Map_UpTrans = np.triu(CMAP, 1).T
            Map_UandL = np.triu(CMAP)
            real_cmap = Map_UandL + Map_UpTrans

            DNCON4_CNN_prediction = real_cmap.reshape(len(p1), Maximum_length*Maximum_length)
            
            global_mse = 0.0
            weighted_mse = 0.0
            if predict_method == 'real_dist':
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                # to binary
                #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
            elif predict_method == 'real_dist_scaled':
                ### convert back to distance 
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0]=0.001
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>=1]=0.999
                #DNCON4_CNN_prediction = 3*np.sqrt((1-DNCON4_CNN_prediction)/DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 10*np.log((1+DNCON4_CNN_prediction)/(1-DNCON4_CNN_prediction))
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                # to binary
                #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
            elif predict_method == 'real_dist_limited':
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                # to binary
                #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
            elif predict_method == 'real_dist_limited2':
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                # to binary
                #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
            elif predict_method == 'real_dist_limited3':
                selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
                global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction, selected_list_label_dist)
                # to binary
                #DNCON4_CNN_prediction = DNCON4_CNN_prediction * (DNCON4_CNN_prediction<=8)
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction<=8]=1
                #DNCON4_CNN_prediction[DNCON4_CNN_prediction>8]=0
                DNCON4_CNN_prediction[DNCON4_CNN_prediction>100] = 100 # incase infinity
                DNCON4_CNN_prediction[DNCON4_CNN_prediction<=0] = 0.001 # incase infinity
                DNCON4_CNN_prediction_dist = np.copy(DNCON4_CNN_prediction)
                DNCON4_CNN_prediction = 1/DNCON4_CNN_prediction # convert to confidence
            
            
            (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = evaluate_prediction_4(p1, DNCON4_CNN_prediction, selected_list_label, 24)
            
            #### calculate  recall and precision for all long range
            #pred_contact = floor_lower_left_to_zero(DNCON4_CNN_prediction, 24)
            #datacount = len(selected_list_label[:, 0])
            #true_contact = floor_lower_left_to_zero(selected_list_label, 24)            
            #pred_contact_flatten = pred_contact.flatten()          
            #true_contact_flatten = true_contact.flatten()
            #precision_all_long = precision_score(true_contact_flatten,pred_contact_flatten)
            #recall_all_long = recall_score(true_contact_flatten,pred_contact_flatten)
            #fscore_all_long = f1_score(true_contact_flatten,pred_contact_flatten)
            
            #val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l,global_mse,weighted_mse,precision_all_long,recall_all_long,fscore_all_long)
            val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l,global_mse,weighted_mse)
            #print("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\tPrecision_all_long\tRecall_all_long\tFscore_all_long\n")
            print("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n")
            print('The best validation accuracy is ',val_acc_history_content)
            best_val_acc_out_inepoch_final = "%s/best_validation_inepoch_final.acc_history" % (CV_dir)
            with open(best_val_acc_out_inepoch_final, "a") as myfile:
                myfile.write(val_acc_history_content)
            DNCON4_CNN_prediction = DNCON4_CNN_prediction.reshape (Maximum_length, Maximum_length)
            # cmap_file = "%s/%s.txt" % (model_predict,key)
            # np.savetxt(cmap_file, DNCON4_CNN_prediction, fmt='%.4f')
            
            
            cmap_file = "%s/%s.txt" % (model_predict_best,key)
            np.savetxt(cmap_file, real_cmap, fmt='%.4f')

            out_gloable_mse += global_mse
            out_weighted_mse += weighted_mse 
            out_avg_pc_l5 += avg_pc_l5 * len(p1)
            out_avg_pc_l2 += avg_pc_l2 * len(p1)
            out_avg_pc_1l += avg_pc_1l * len(p1)
            out_avg_acc_l5 += avg_acc_l5 * len(p1)
            out_avg_acc_l2 += avg_acc_l2 * len(p1)
            out_avg_acc_1l += avg_acc_1l * len(p1)
            #out_precision_all_long += precision_all_long
            #out_recall_all_long += recall_all_long
            #out_fscore_all_long += fscore_all_long
            
            step_num += 1
        print ('step_num=', step_num)
        all_num = len(selected_list)
        out_gloable_mse /= all_num
        out_weighted_mse /= all_num
        out_avg_pc_l5 /= all_num
        out_avg_pc_l2 /= all_num
        out_avg_pc_1l /= all_num
        out_avg_acc_l5 /= all_num
        out_avg_acc_l2 /= all_num
        out_avg_acc_1l /= all_num
        #out_precision_all_long /= all_num
        #out_recall_all_long /= all_num
        #out_fscore_all_long /= all_num
        val_acc_history_content = "%i\t%i\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (interval_len,epoch,epoch_inside,out_avg_pc_l5,out_avg_pc_l2,out_avg_pc_1l,
            out_avg_acc_l5,out_avg_acc_l2,out_avg_acc_1l, out_gloable_mse, out_weighted_mse)
        print('The final val_acc_history_content: ',val_acc_history_content)
        best_val_acc_out_final = "%s/best_validation_final.acc_history" % (CV_dir)
        with open(best_val_acc_out_final, "a") as myfile:
                    myfile.write(val_acc_history_content)  




def DNCON4_predict_testing(evalist,model_file,model_weight_out_best,cmap_file,opt,path_of_lists, path_of_Y, path_of_X, Maximum_length,dist_string, reject_fea_file='None',
    initializer = "he_normal", loss_function = "weighted_BCE", weight_p=1.0, weight_n=1.0,  list_sep_flag=False,  if_use_binsize = False): 

    import numpy as np
    Train_data_keys = dict()
    Train_targets_keys = dict()
    
    json_file_model = open(model_file, 'r')
    loaded_model_json = json_file_model.read()
    json_file_model.close()    
    DNCON4_CNN = model_from_json(loaded_model_json, custom_objects={'K_max_pooling1d': K_max_pooling1d})        
            
    rerun_flag=0
    if os.path.exists(model_weight_out_best):
        print("######## Loading existing weights ",model_weight_out_best)
        DNCON4_CNN.load_weights(model_weight_out_best)
        rerun_flag = 1
    else:
        print("failed to find weight ",model_weight_out_best) 
        exit(-1)  
    
    #predict_method has three value : bin_class, mul_class, real_dist
    predict_method = 'bin_class'
    if loss_function == 'weighted_BCE':
        predict_method = 'bin_class'
        path_of_Y_train = path_of_Y + '/bin_class/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        if weight_p <= 1:
            weight_n = 1.0 - weight_p
        loss_function = _weighted_binary_crossentropy(weight_p, weight_n)
    elif loss_function == 'weighted_CCE':
        predict_method = 'mul_class'
        loss_function = _weighted_categorical_crossentropy(weight_p)
    elif loss_function == 'weighted_MSE':
        predict_method = 'real_dist'
        path_of_Y_train = path_of_Y + '/real_dist/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = _weighted_mean_squared_error(1)
    elif loss_function == 'sigmoid_MSE':
        predict_method = 'real_dist_scaled'
        path_of_Y_train = path_of_Y + '/real_dist/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'mean_squared_error'
    elif loss_function == 'categorical_crossentropy':
        predict_method = 'mul_class'
        path_of_Y_train = path_of_Y + '/dist_map/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'categorical_crossentropy'
    else:
        predict_method = 'real_dist'
        path_of_Y_train = path_of_Y + '/real_dist/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = loss_function
    print("Setting predict_method to ",predict_method)
    print("Setting loss function to ",loss_function)

    DNCON4_CNN.compile(loss=loss_function, metrics=['acc'], optimizer=opt)


    length_dict = {}
    with open(path_of_lists + 'L.txt') as f:
      for line in f:
        cols = line.strip().split()
        length_dict[cols[0]] = int(cols[1])
    te_l = {}
    with open(evalist) as f:
      for line in f:
        te_l[line.strip()] = length_dict[line.strip()]
    print('Total Number of Training dataset = ',str(len(te_l)))

    # callbacks=[reduce_lr]
    train_avg_acc_l5_best = 0 
    val_avg_acc_l5_best = 0
    min_seq_sep = 0
    lr_decay = False
    train_loss_last = 1e32
    train_loss_list = []
    evalu_loss_list = []
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    selected_list = subset_pdb_dict(te_l,   0, 500, 5000, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
    for key in selected_list:
        print('saving cmap of %s\n'%(key))
        value = selected_list[key]
        single_dict={key:value}
        if if_use_binsize:
            Maximum_length = 320
        else:
            Maximum_length = value
        # print(single_dict)
        selected_list_2D = get_x_2D_from_this_list(single_dict, path_of_X, Maximum_length, dist_string, reject_fea_file, value)
        print("selected_list_2D.shape: ",selected_list_2D.shape)
        print('Loading label sets..')
        selected_list_label = get_y_from_this_list(single_dict, path_of_Y, min_seq_sep, Maximum_length, dist_string)
        DNCON4_CNN.load_weights(model_weight_out_best)

        DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_2D], batch_size= 1)
        DNCON4_CNN_prediction = DNCON4_CNN_prediction.reshape (Maximum_length, Maximum_length)
        np.savetxt(cmap_file, DNCON4_CNN_prediction, fmt='%.4f')
        # history_loss_file = CV_dir+"/train_loss.history"
