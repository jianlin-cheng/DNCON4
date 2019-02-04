# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:47:26 2017

@author: Jie Hou
"""
import os

from Model_construct import *
from DNCON_lib import *

from Model_construct import _weighted_binary_crossentropy, _weighted_categorical_crossentropy, _weighted_binary_crossentropy_shield

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
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D, Convolution2D
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from random import randint


"""

import sys
import os
from shutil import copyfile
import platform
## test only
GLOBAL_PATH= '/storage/htc/bdm/jh7x3/DNCON4/architecture/CNN_arch'
print GLOBAL_PATH
sys.path.insert(0, GLOBAL_PATH+'/lib/')
from Model_construct import DNCON4_with_paras

win_array = [6]
feature_1D_num=25
feature_2D_num=18
sequence_length=220
use_bias=True
hidden_type='sigmoid'
nb_filters=5
nb_layers=6
opt='nadam'
DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt)
"""




def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

def DNCON4_1d2dconv_train_win_filter_layer_opt_fast(data_all_dict_padding,testdata_all_dict_padding,CV_dir,feature_dir,model_prefix,epoch_outside,epoch_inside,interval_len,seq_end,win_array,use_bias,hidden_type,nb_filters,nb_layers,opt,lib_dir, batch_size_train,path_of_lists,path_of_Y, path_of_X,Maximum_length,dist_string,reject_fea_file='None'): 
    start=0
    end=seq_end
    import numpy as np
    Train_data_keys = dict()
    Train_targets_keys = dict()
    # Test_data_keys = dict()
    # Test_targets_keys = dict()
    
    feature_num=0; # the number of features for each residue
    for key in sorted(data_all_dict_padding.keys()):
        if key <start: # run first model on 100 at most
            continue
        if key > end: # run first model on 100 at most
            continue
        print('### Loading sequence length :', key)
        seq_len=key
        trainfeaturedata = data_all_dict_padding[key]
        train_label_all = []
        train_1D_fea_all = []
        train_2D_fea_all = []
        for i in range(0,len(trainfeaturedata)):
          train_labels = trainfeaturedata[i][0] ## (seq_len*seq_len,)
          train_1D_feature = trainfeaturedata[i][1] ## (seq_len, 1d_fea_num)
          train_2D_feature = trainfeaturedata[i][2] ## (seq_len, seq_len, 2d_fea_num)
          feature_1D_num=train_1D_feature.shape[1]
          feature_2D_num=train_2D_feature.shape[2]  
          train_label_all.append(train_labels)
          train_1D_fea_all.append(train_1D_feature)
          train_2D_fea_all.append(train_2D_feature)
        
        
        # if seq_len in testdata_all_dict_padding:
        #     testfeaturedata = testdata_all_dict_padding[seq_len]
        #     #print "Loading test dataset "
        # else:
        #     testfeaturedata = trainfeaturedata
        
        # test_label_all = []
        # test_1D_fea_all = []
        # test_2D_fea_all = []
        # for i in range(0,len(testfeaturedata)):
        #   test_labels = testfeaturedata[i][0] ## (seq_len*seq_len,)
        #   test_1D_feature = testfeaturedata[i][1] ## (seq_len, 1d_fea_num)
        #   test_2D_feature = testfeaturedata[i][2] ## (seq_len, seq_len, 2d_fea_num)
        #   test_label_all.append(test_labels)
        #   test_1D_fea_all.append(test_1D_feature)
        #   test_2D_fea_all.append(test_2D_feature)
          
        train_label_all_array = np.asarray(train_label_all) #(21, 48400)
        train_1D_fea_all_array = np.asarray(train_1D_fea_all) #(21, 220, 22)
        train_2D_fea_all_array = np.asarray(train_2D_fea_all) #(21, 220, 220, 18)  
        train_2D_fea_all_array = train_2D_fea_all_array.reshape(train_2D_fea_all_array.shape[0],train_2D_fea_all_array.shape[1],train_2D_fea_all_array.shape[2],train_2D_fea_all_array.shape[3])     #(21, 220*220, 18)  
          
        # test_label_all_array = np.asarray(test_label_all) #(21, 48400)
        # test_1D_fea_all_array = np.asarray(test_1D_fea_all) #(21, 220, 22)
        # test_2D_fea_all_array = np.asarray(test_2D_fea_all) #(21, 220, 220, 18)        
        # test_2D_fea_all_array = test_2D_fea_all_array.reshape(test_2D_fea_all_array.shape[0],test_2D_fea_all_array.shape[1],test_2D_fea_all_array.shape[2],test_2D_fea_all_array.shape[3])  #(21, 220*220, 18)  
        
        sequence_length = seq_len
               
        
        if seq_len in Train_data_keys:
            raise Exception("Duplicate seq length %i in Train list, since it has been combined when loading data " % seq_len)
        else:
            Train_data_keys[seq_len]=[train_1D_fea_all_array,train_2D_fea_all_array]
            
        if seq_len in Train_targets_keys:
            raise Exception("Duplicate seq length %i in Train list, since it has been combined when loading data " % seq_len)
        else:
            Train_targets_keys[seq_len]=train_label_all_array        
        #processing test data 
        # if seq_len in Test_data_keys:
        #     raise Exception("Duplicate seq length %i in Test list, since it has been combined when loading data " % seq_len)
        # else:
        #     Test_data_keys[seq_len]=[test_1D_fea_all_array,test_2D_fea_all_array]
        
        # if seq_len in Test_targets_keys:
        #     raise Exception("Duplicate seq length %i in Test list, since it has been combined when loading data " % seq_len)
        # else:
        #     Test_targets_keys[seq_len]=test_label_all_array
 
    train_avg_acc_l5_best = 0 
    val_avg_acc_l5_best = 0
        
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    chkdirs(train_acc_history_out)     
    with open(train_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tAvg_Precision_l5\tAvg_Precision_l2\tAvg_Precision_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\n")
      
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    chkdirs(val_acc_history_out)     
    with open(val_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tAvg_Precision_l5\tAvg_Precision_l2\tAvg_Precision_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\n")
    
    #Loading Validation data into Memory
    #Maximum_length=300 # 800 will get memory error
    tr_l, tr_n, tr_e, te_l, te_n, te_e = build_dataset_dictionaries(path_of_lists)
    # Make combined dictionaries as well
    all_l = te_l.copy()
    all_n = te_n.copy()
    all_e = te_e.copy()
    all_l.update(tr_l)
    all_n.update(tr_n)
    all_e.update(tr_e)
    print('Total Number of Training and Test dataset = ',str(len(all_l)))
    
    sys.stdout.flush()
    print('Load all test data into memory..',end='')
    selected_list = subset_pdb_dict(te_l,   0, Maximum_length, Maximum_length, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
    print('Loading data sets ..',end='')
    (selected_list_1D,selected_list_2D) = get_x_1D_2D_from_this_list(selected_list, path_of_X, Maximum_length,dist_string,reject_fea_file)
    print("selected_list_1D.sum: ",np.sum(selected_list_1D))
    print("selected_list_2D.sum: ",np.sum(selected_list_2D))
    print("selected_list_1D.shape: ",selected_list_1D.shape)
    print("selected_list_2D.shape: ",selected_list_2D.shape)
    print('Loading label sets..')
    selected_list_label = get_y_from_this_list(selected_list, path_of_Y, 24, Maximum_length, dist_string)
    feature_1D_num_vali = selected_list_1D.shape[2]
    feature_2D_num_vali = selected_list_2D.shape[3]
    sequence_length = selected_list_1D.shape[1]

    ### Define the model 
    model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
    model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
    model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)

    for epoch in range(0,epoch_outside):
        print("\n############ Running epoch ", epoch)
        for key in sorted(data_all_dict_padding.keys()):
            if key <start: # run first model on 100 at most
                continue
            if key > end: # run first model on 100 at most
                continue
            print('### Loading sequence length :', key)
            seq_len=key
            train_featuredata_all=Train_data_keys[seq_len]
            train_targets=Train_targets_keys[seq_len]
            train_1D_fea_all_array=train_featuredata_all[0]
            train_2D_fea_all_array=train_featuredata_all[1]
            
            # test_featuredata_all=Test_data_keys[seq_len]
            # test_targets=Test_targets_keys[seq_len]
            # test_1D_fea_all_array=test_featuredata_all[0]
            # test_2D_fea_all_array=test_featuredata_all[1]
            
            print("Train 1D shape: ",train_1D_fea_all_array.shape, " in outside epoch ", epoch)
            print("Train 2D shape: ",train_2D_fea_all_array.shape, " in outside epoch ", epoch)
            # print("Test 1D shape: ",test_1D_fea_all_array.shape, " in outside epoch ", epoch)
            # print("Test 2D shape: ",test_2D_fea_all_array.shape, " in outside epoch ", epoch)
            
            
            ## because the current model need batch size as parameter, so if the number of samples in training is less than batch size, it will report error when training, so
            ######## expand training dataset for batch training
            batch_size_train_new = batch_size_train
            if train_targets.shape[0] < batch_size_train:
              print("Setting batch size from ",batch_size_train, " to ",train_targets.shape[0])
              batch_size_train_new = train_targets.shape[0]
            else:
              factor = int(train_targets.shape[0] / batch_size_train)  # 7/5 = 1
              num_to_expand = (factor+1)*batch_size_train - train_targets.shape[0]
              random_to_pick = np.random.randint(0,train_targets.shape[0],num_to_expand)
              train_1D_fea_all_array_expand = np.zeros((train_targets.shape[0]+num_to_expand,train_1D_fea_all_array.shape[1],train_1D_fea_all_array.shape[2]))
              train_1D_fea_all_array_expand[:train_1D_fea_all_array.shape[0],:train_1D_fea_all_array.shape[1],:train_1D_fea_all_array.shape[2]] = train_1D_fea_all_array
              
              train_2D_fea_all_array_expand = np.zeros((train_targets.shape[0]+num_to_expand,train_2D_fea_all_array.shape[1],train_2D_fea_all_array.shape[2],train_2D_fea_all_array.shape[3]))
              train_2D_fea_all_array_expand[:train_2D_fea_all_array.shape[0],:train_2D_fea_all_array.shape[1],:train_2D_fea_all_array.shape[2],:train_2D_fea_all_array.shape[3]] = train_2D_fea_all_array
              
              train_label_all_array_expand = np.zeros((train_targets.shape[0]+num_to_expand,train_targets.shape[1]))
              train_label_all_array_expand[:train_targets.shape[0],:train_targets.shape[1]] = train_targets
              for indx in range(0,len(random_to_pick)):
                label_select = train_targets[random_to_pick[indx],:]
                train_1D_fea_select = train_1D_fea_all_array[random_to_pick[indx],:,:]
                train_2D_fea_select = train_2D_fea_all_array[random_to_pick[indx],:,:,:]
                
                train_label_all_array_expand[train_targets.shape[0]+indx,:] = label_select
                train_1D_fea_all_array_expand[train_1D_fea_all_array.shape[0]+indx,:,:] = train_1D_fea_select
                train_2D_fea_all_array_expand[train_2D_fea_all_array.shape[0]+indx,:,:,:] = train_2D_fea_select
              
              train_1D_fea_all_array = train_1D_fea_all_array_expand
              train_2D_fea_all_array = train_2D_fea_all_array_expand
              train_targets = train_label_all_array_expand
            
            print("The expanded train label size: ",train_targets.shape)
            print("The expanded train 1D fea size: ",train_1D_fea_all_array.shape)
            print("The expanded train 2D fea size: ",train_2D_fea_all_array.shape)
            
            
            sequence_length = seq_len
            
            print("######## Setting initial model based on length ",sequence_length)
            # ktop_node is the length of input proteins
            if model_prefix == 'DNCON4_1d2dCNN':
                print(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
                DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
            elif model_prefix == 'DNCON4_1d2dCRMN':
                DNCON4_CNN = DeepCRMN_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
            elif model_prefix == 'DNCON4_1d2dFRAC':
                DNCON4_CNN = DeepFracNet_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
            elif model_prefix == 'DNCON4_1d2dINCEP':
                DNCON4_CNN = DeepInception_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
            elif model_prefix == 'DNCON4_1d2dRCNN':
                DNCON4_CNN = DeepCovRCNN_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
            elif model_prefix == 'DNCON4_1d2dRES':
                DNCON4_CNN = DeepResnet_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
            elif model_prefix == 'DNCON4_1d2dRESATT':
                DNCON4_CNN = DeepCovResAtt_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
            elif model_prefix == 'DNCON4_1d2dRCINCEP':
                DNCON4_CNN = DNCON4_RCIncep_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
            else:
                DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
            
            rerun_flag=0
            # DNCON4_CNN = multi_gpu_model(DNCON4_CNN, gpus=2)
            if key <= 150 and epoch == 0:
                if os.path.exists(model_weight_out_best):
                    print("######## Loading existing weights ",model_weight_out_best)
                    rerun_flag=1
                    DNCON4_CNN.load_weights(model_weight_out_best)
            else:
                if os.path.exists(model_weight_out):
                    print("######## Loading existing weights ",model_weight_out)
                    DNCON4_CNN.load_weights(model_weight_out)
                else:
                    print("######## Setting initial weights")   
            
            DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)

            if key <= 150 and epoch == 0 and rerun_flag == 0:
              DNCON4_CNN.fit([train_1D_fea_all_array,train_2D_fea_all_array], train_targets, batch_size= batch_size_train_new, epochs=10, verbose=1)
            else:
              DNCON4_CNN.fit([train_1D_fea_all_array,train_2D_fea_all_array], train_targets, batch_size= batch_size_train_new, epochs=epoch_inside, verbose=1)
            DNCON4_CNN.save_weights(model_weight_out)
            # serialize model to JSON
            # model_json = DNCON4_CNN.to_json()
            # print("Saved model to disk")
            # with open(model_out, "w") as json_file:
            #     json_file.write(model_json)
            gc.collect()
            
        
        ### save models
        model_weight_out_inepoch = "%s/model-train-weight-%s-epoch%i.h5" % (CV_dir,model_prefix,epoch)
        DNCON4_CNN.save_weights(model_weight_out_inepoch)
        # weight_array= DNCON4_CNN.get_weights()
        # weight_filename="%s/model-train-weight-%s-epoch%i.txt" % (CV_dir,model_prefix,epoch)
        # fileObject = open(weight_filename, 'w')
        # for i in range(len(weight_array)):
        #     fileObject.write(str(weight_array[i])+'\n')
        # fileObject.close()

        ##### running validation
        print("Now evaluate for epoch ",epoch)
        #dist_string = '80'
        #path_of_lists = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/lists-test-train/'
        #path_of_Y         = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/'
        #path_of_X         = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/'
        #Maximum_length=300 # 800 will get memory error
        # tr_l, tr_n, tr_e, te_l, te_n, te_e = build_dataset_dictionaries(path_of_lists)
        # # Make combined dictionaries as well
        # all_l = te_l.copy()
        # all_n = te_n.copy(
        # all_e = te_e.copy()
        # all_l.update(tr_l)
        # all_n.update(tr_n)
        # all_e.update(tr_e)
        # print('Total Number of Training and Test dataset = ',str(len(all_l)))
        
        # sys.stdout.flush()
        # print('Load all test data into memory..',end='')
        # selected_list = subset_pdb_dict(te_l,   0, Maximum_length, Maximum_length, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
        # print('Loading data sets ..',end='')
        # (selected_list_1D,selected_list_2D) = get_x_1D_2D_from_this_list(selected_list, path_of_X, Maximum_length,dist_string)
        # print("selected_list_1D.sum: ",np.sum(selected_list_1D))
        # print("selected_list_2D.sum: ",np.sum(selected_list_2D))
        # print("selected_list_1D.shape: ",selected_list_1D.shape)
        # print("selected_list_2D.shape: ",selected_list_2D.shape)
        # print('Loading label sets..')
        # selected_list_label = get_y_from_this_list(selected_list, path_of_Y, 24, Maximum_length, dist_string)
        # feature_1D_num_vali = selected_list_1D.shape[2]
        # feature_2D_num_vali = selected_list_2D.shape[3]
        # sequence_length = selected_list_1D.shape[1]
        if model_prefix == 'DNCON4_1d2dCNN':
            DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num_vali,feature_2D_num_vali,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
        elif model_prefix == 'DNCON4_1d2dCRMN':
            DNCON4_CNN = DeepCRMN_with_paras(win_array,feature_1D_num_vali,feature_2D_num_vali,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
        elif model_prefix == 'DNCON4_1d2dFRAC':
            DNCON4_CNN = DeepFracNet_with_paras(win_array,feature_1D_num_vali,feature_2D_num_vali,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
        elif model_prefix == 'DNCON4_1d2dINCEP':
            DNCON4_CNN = DeepInception_with_paras(win_array,feature_1D_num_vali,feature_2D_num_vali,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
        elif model_prefix == 'DNCON4_1d2dRCNN':
            DNCON4_CNN = DeepCovRCNN_with_paras(win_array,feature_1D_num_vali,feature_2D_num_vali,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
        elif model_prefix == 'DNCON4_1d2dRES':
            DNCON4_CNN = DeepResnet_with_paras(win_array,feature_1D_num_vali,feature_2D_num_vali,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
        elif model_prefix == 'DNCON4_1d2dRESATT':
            DNCON4_CNN = DeepCovResAtt_with_paras(win_array,feature_1D_num_vali,feature_2D_num_vali,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
        elif model_prefix == 'DNCON4_1d2dRCINCEP':
            DNCON4_CNN = DNCON4_RCIncep_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
        else:
            DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num_vali,feature_2D_num_vali,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)

        # DNCON4_CNN = multi_gpu_model(DNCON4_CNN, gpus=2)
        DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
        DNCON4_CNN.load_weights(model_weight_out_inepoch)
        DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_1D,selected_list_2D], batch_size= 1)
        (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = evaluate_prediction(selected_list, all_n, all_e, DNCON4_CNN_prediction, selected_list_label, 24)
        
        
        val_acc_history_content = "%i\t%i\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (interval_len,epoch,epoch_inside,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l)
        with open(val_acc_history_out, "a") as myfile:
                    myfile.write(val_acc_history_content)  
        
        if avg_acc_l5 >= val_avg_acc_l5_best:
            val_avg_acc_l5_best = avg_acc_l5 
            score_imed = "Accuracy L5 of Val: %.4f\t\n" % (val_avg_acc_l5_best)
            model_json = DNCON4_CNN.to_json()
            print("Saved model to disk")
            with open(model_out, "w") as json_file:
                json_file.write(model_json)
            print("Saved best weight to disk, ", score_imed)
            DNCON4_CNN.save_weights(model_weight_out_best)
        print('The val accuracy is ',val_acc_history_content)

        #clear memory
        K.clear_session()
        tf.reset_default_graph()

    
    print("Training finished, best validation acc = ",val_avg_acc_l5_best)
    return val_avg_acc_l5_best

def DNCON4_1d2dconv_train_win_filter_layer_opt_fast_2Donly(data_all_dict_padding,testdata_all_dict_padding,CV_dir,feature_dir,model_prefix,epoch_outside,epoch_inside,interval_len,seq_end,win_array,use_bias,hidden_type,nb_filters,nb_layers,opt,lib_dir, batch_size_train,path_of_lists,path_of_Y, path_of_X,Maximum_length,dist_string, reject_fea_file='None'): 
    start=0
    end=seq_end
    import numpy as np
    Train_data_keys = dict()
    Train_targets_keys = dict()
    Test_data_keys = dict()
    Test_targets_keys = dict()
    
    feature_num=0; # the number of features for each residue
    for key in sorted(data_all_dict_padding.keys()):
        if key <start: # run first model on 100 at most
            continue
        if key > end: # run first model on 100 at most
            continue
        print('### Loading sequence length :', key)
        seq_len=key
        trainfeaturedata = data_all_dict_padding[key]
        train_label_all = []
        train_2D_fea_all = []
        for i in range(0,len(trainfeaturedata)):
          train_labels = trainfeaturedata[i][0] ## (seq_len*seq_len,)
          train_2D_feature = trainfeaturedata[i][1] ## (seq_len, seq_len, 2d_fea_num)
          feature_2D_num=train_2D_feature.shape[2]  ##2
          train_label_all.append(train_labels)
          train_2D_fea_all.append(train_2D_feature)
        
        
        if seq_len in testdata_all_dict_padding:
            testfeaturedata = testdata_all_dict_padding[seq_len]
            #print "Loading test dataset "
        else:
            testfeaturedata = trainfeaturedata
            print("\n\n##Warning: Setting training dataset as testing dataset \n\n")
        
        test_label_all = []
        test_2D_fea_all = []
        for i in range(0,len(testfeaturedata)):
          test_labels = testfeaturedata[i][0] ## (seq_len*seq_len,)
          test_2D_feature = testfeaturedata[i][1] ## (seq_len, seq_len, 2d_fea_num)
          test_label_all.append(test_labels)
          test_2D_fea_all.append(test_2D_feature)
          
        train_label_all_array = np.asarray(train_label_all) #(21, 48400)
        train_2D_fea_all_array = np.asarray(train_2D_fea_all) #(21, 220, 220, 18)  
        train_2D_fea_all_array = train_2D_fea_all_array.reshape(train_2D_fea_all_array.shape[0],train_2D_fea_all_array.shape[1],train_2D_fea_all_array.shape[2],train_2D_fea_all_array.shape[3])     #(21, 220*220, 18)  
          
        test_label_all_array = np.asarray(test_label_all) #(21, 48400)
        test_2D_fea_all_array = np.asarray(test_2D_fea_all) #(21, 220, 220, 18)        
        test_2D_fea_all_array = test_2D_fea_all_array.reshape(test_2D_fea_all_array.shape[0],test_2D_fea_all_array.shape[1],test_2D_fea_all_array.shape[2],test_2D_fea_all_array.shape[3])  #(21, 220*220, 18)  
        
        sequence_length = seq_len
               
        
        if seq_len in Train_data_keys:
            raise Exception("Duplicate seq length %i in Train list, since it has been combined when loading data " % seq_len)
        else:
            Train_data_keys[seq_len]=[train_2D_fea_all_array]
            
        if seq_len in Train_targets_keys:
            raise Exception("Duplicate seq length %i in Train list, since it has been combined when loading data " % seq_len)
        else:
            Train_targets_keys[seq_len]=train_label_all_array        
        #processing test data 
        if seq_len in Test_data_keys:
            raise Exception("Duplicate seq length %i in Test list, since it has been combined when loading data " % seq_len)
        else:
            Test_data_keys[seq_len]=[test_2D_fea_all_array]
        
        if seq_len in Test_targets_keys:
            raise Exception("Duplicate seq length %i in Test list, since it has been combined when loading data " % seq_len)
        else:
            Test_targets_keys[seq_len]=test_label_all_array
 
    train_avg_acc_l5_best = 0 
    val_avg_acc_l5_best = 0
        
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    chkdirs(train_acc_history_out)     
    with open(train_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tAvg_Precision_l5\tAvg_Precision_l2\tAvg_Precision_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\n")
      
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    chkdirs(val_acc_history_out)     
    with open(val_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tAvg_Precision_l5\tAvg_Precision_l2\tAvg_Precision_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\n")
    
    tr_l, tr_n, tr_e, te_l, te_n, te_e = build_dataset_dictionaries(path_of_lists)
    # Make combined dictionaries as well
    all_l = te_l.copy()
    all_n = te_n.copy()
    all_e = te_e.copy()
    all_l.update(tr_l)
    all_n.update(tr_n)
    all_e.update(tr_e)
    print('Total Number of Training and Test dataset = ',str(len(all_l)))
    
    # sys.stdout.flush()
    # print('Load all test data into memory..',end='')
    # selected_list = subset_pdb_dict(te_l,   0, Maximum_length, Maximum_length, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
    # print('Loading data sets ..',end='')
    # print(selected_list, path_of_X, Maximum_length,dist_string)
    # selected_list_2D = get_x_2D_from_this_list(selected_list, path_of_X, Maximum_length,dist_string, reject_fea_file)
    # print("selected_list_2D.shape: ",selected_list_2D.shape)
    # print('Loading label sets..')
    # selected_list_label = get_y_from_this_list(selected_list, path_of_Y, 24, Maximum_length, dist_string)
    # feature_2D_num_vali = selected_list_2D.shape[3]      

    model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
    model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
    model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)
    print("######## Setting initial model based on length ",sequence_length)
    ## ktop_node is the length of input proteins
    if model_prefix == 'DNCON4_1d2dconv':
        DNCON4_CNN = DNCON4_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt)
    elif model_prefix == 'DNCON4_2dINCEP':
        DNCON4_CNN = DeepInception_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt)
    elif model_prefix == 'DNCON4_2dRES':
        DNCON4_CNN = DeepResnet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt)
    elif model_prefix == 'DNCON4_2dRCNN':
        DNCON4_CNN = DeepCovRCNN_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt)
    else:
        DNCON4_CNN = DNCON4_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt)

    rerun_flag=0
    # with tf.device("/cpu:0"):
    #     DNCON4_CNN = multi_gpu_model(DNCON4_CNN, gpus=2)

    if os.path.exists(model_weight_out):
        print("######## Loading existing weights ",model_weight_out)
        rerun_flag=1
        DNCON4_CNN.load_weights(model_weight_out)
    else:
        print("######## Setting initial weights")   
    
    DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
    for epoch in range(0,epoch_outside):
        print("\n############ Running epoch ", epoch)
        for key in sorted(data_all_dict_padding.keys()):
            if key <start: # run first model on 100 at most
                continue
            if key > end: # run first model on 100 at most
                continue
            print('### Loading sequence length :', key)
            seq_len=key
            train_featuredata_all=Train_data_keys[seq_len]
            train_targets=Train_targets_keys[seq_len]
            train_2D_fea_all_array=train_featuredata_all[0]
            
            test_featuredata_all=Test_data_keys[seq_len]
            test_targets=Test_targets_keys[seq_len]
            test_2D_fea_all_array=test_featuredata_all[0]
            
            print("Train 2D shape: ",train_2D_fea_all_array.shape, " in outside epoch ", epoch)
            print("Test 2D shape: ",test_2D_fea_all_array.shape, " in outside epoch ", epoch)
            
            
            ## because the current model need batch size as parameter, so if the number of samples in training is less than batch size, it will report error when training, so
            ######## expand training dataset for batch training
            batch_size_train_new = batch_size_train
            if train_targets.shape[0] < batch_size_train:
              print("Setting batch size from ",batch_size_train, " to ",train_targets.shape[0])
              batch_size_train_new = train_targets.shape[0]
            else:
              factor = int(train_targets.shape[0] / batch_size_train)  # 7/5 = 1
              num_to_expand = (factor+1)*batch_size_train - train_targets.shape[0]
              random_to_pick = np.random.randint(0,train_targets.shape[0],num_to_expand)
              train_2D_fea_all_array_expand = np.zeros((train_targets.shape[0]+num_to_expand,train_2D_fea_all_array.shape[1],train_2D_fea_all_array.shape[2],train_2D_fea_all_array.shape[3]))
              train_2D_fea_all_array_expand[:train_2D_fea_all_array.shape[0],:train_2D_fea_all_array.shape[1],:train_2D_fea_all_array.shape[2],:train_2D_fea_all_array.shape[3]] = train_2D_fea_all_array
              
              train_label_all_array_expand = np.zeros((train_targets.shape[0]+num_to_expand,train_targets.shape[1]))
              train_label_all_array_expand[:train_targets.shape[0],:train_targets.shape[1]] = train_targets
              for indx in range(0,len(random_to_pick)):
                label_select = train_targets[random_to_pick[indx],:]
                train_2D_fea_select = train_2D_fea_all_array[random_to_pick[indx],:,:,:]
                
                train_label_all_array_expand[train_targets.shape[0]+indx,:] = label_select
                train_2D_fea_all_array_expand[train_2D_fea_all_array.shape[0]+indx,:,:,:] = train_2D_fea_select
              
              train_2D_fea_all_array = train_2D_fea_all_array_expand
              train_targets = train_label_all_array_expand
            
            print("The expanded train label size: ",train_targets.shape)
            print("The expanded train 2D fea size: ",train_2D_fea_all_array.shape)      
            
            ### Define the model 

            
            sequence_length = seq_len


            
            # print('train_targets', train_targets.shape)
            # print('lens', train_targets.shape[0])
            train_targets = train_targets.reshape(train_targets.shape[0], train_2D_fea_all_array.shape[1], train_2D_fea_all_array.shape[1], 1)
            if key <= 150 and epoch == 0 and rerun_flag == 0:
                DNCON4_CNN.fit([train_2D_fea_all_array], train_targets, batch_size= batch_size_train_new, epochs=20, verbose=1)
            else:
                DNCON4_CNN.fit([train_2D_fea_all_array], train_targets, batch_size= batch_size_train_new, epochs=epoch_inside, verbose=1)

            DNCON4_CNN.save_weights(model_weight_out)
            # serialize model to JSON
            # model_json = DNCON4_CNN.to_json()
            # print("Saved model to disk")
            # with open(model_out, "w") as json_file:
            #     json_file.write(model_json)
        
        
        ### save models
        model_weight_out_inepoch = "%s/model-train-weight-%s-epoch%i.h5" % (CV_dir,model_prefix,epoch)
        DNCON4_CNN.save_weights(model_weight_out_inepoch)
        ##### running validation
        print("Now evaluate for epoch ",epoch)
        

        #dist_string = '80'
        #path_of_lists = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/lists-test-train/'
        #path_of_Y         = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/'
        #path_of_X         = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/'
        #Maximum_length=300 # 800 will get memory error

        # sys.stdout.flush()
        # print('Load all test data into memory..',end='')
        selected_list = subset_pdb_dict(te_l,   0, Maximum_length, Maximum_length, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
        print('Loading data sets ..',end='')
        selected_list_2D = get_x_2D_from_this_list(selected_list, path_of_X, Maximum_length,dist_string, reject_fea_file)
        print("selected_list_2D.shape: ",selected_list_2D.shape)
        print('Loading label sets..')
        selected_list_label = get_y_from_this_list(selected_list, path_of_Y, 24, Maximum_length, dist_string)
        feature_2D_num = selected_list_2D.shape[3]

        # if model_prefix == 'DNCON4_1d2dconv':
        #     DNCON4_CNN = DNCON4_with_paras_2D(win_array,feature_2D_num_vali,use_bias,hidden_type,nb_filters,nb_layers,opt)
        # elif model_prefix == 'DNCON4_2dINCEP':
        #     DNCON4_CNN = DeepInception_with_paras_2D(win_array,feature_2D_num_vali,use_bias,hidden_type,nb_filters,nb_layers,opt)
        # elif model_prefix == 'DNCON4_2dRES':
        #         DNCON4_CNN = DeepResnet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt)
        # elif model_prefix == 'DNCON4_2dRCNN':
        #         DNCON4_CNN = DeepCovRCNN_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt)
        # else:
        #     DNCON4_CNN = DNCON4_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt)
        # DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
        # DNCON4_CNN.load_weights(model_weight_out)

        ######
        DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_2D], batch_size= 1)
        ##flatteng
        DNCON4_CNN_prediction = DNCON4_CNN_prediction.reshape(len(selected_list), Maximum_length*Maximum_length)
        (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = evaluate_prediction(selected_list, all_n, all_e, DNCON4_CNN_prediction, selected_list_label, 24)
        
        val_acc_history_content = "%i\t%i\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (interval_len,epoch,epoch_inside,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l)
        with open(val_acc_history_out, "a") as myfile:
                    myfile.write(val_acc_history_content)  
        
        if avg_acc_l5 >= val_avg_acc_l5_best:
            val_avg_acc_l5_best = avg_acc_l5 
            score_imed = "Accuracy L5 of Val: %.4f\t\n" % (val_avg_acc_l5_best)
            model_json = DNCON4_CNN.to_json()
            print("Saved model to disk")
            with open(model_out, "w") as json_file:
                json_file.write(model_json)
            print("Saved best weight to disk, ", score_imed)
            DNCON4_CNN.save_weights(model_weight_out_best)
        print('The val accuracy is ',val_acc_history_content)
        
    
        #clear memory
        # K.clear_session()
        # tf.reset_default_graph()

    
    print("Training finished, best validation acc = ",val_avg_acc_l5_best)
    return val_avg_acc_l5_best
# dist_string = '80'interval
def generate_data_from_file(path_of_lists, feature_dir, min_seq_sep,dist_string, batch_size, reject_fea_file='None', child_list_index=0, list_sep_flag=False, dataset_select='train'):
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
    else:
        dataset_list = build_dataset_dictionaries_test(path_of_lists)

    if (list_sep_flag == False):
        training_dict = subset_pdb_dict(dataset_list, 0, 500, 5000, 'random') #can be random ordered   
        training_list = list(training_dict.keys())
        training_lens = list(training_dict.values())
        all_data_num = len(training_dict)
        loopcount = all_data_num // int(batch_size)
        print('all_num=',all_data_num)
        print('loopcount=',loopcount)
    else:
        training_dict = subset_pdb_dict(dataset_list, 0, 500, 5000, 'ordered') #can be random ordered
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
        max_pdb_lens = max(batch_list_len)
        # if max_pdb_lens <= 300:
        #     max_pdb_lens=300
        # elif max_pdb_lens <=300 and max_pdb_lens > 150:
        #     max_pdb_lens=300
        # if max_pdb_lens <= 100:
        #     max_pdb_lens=100
        # elif max_pdb_lens <=200 and max_pdb_lens > 100:
        #     max_pdb_lens=200
        # elif max_pdb_lens <=300 and max_pdb_lens > 200:
        #     max_pdb_lens=300
        # else:
        #     max_pdb_lens = max(batch_list_len)
        data_all_dict = dict()
        batch_X=[]
        batch_Y=[]
        for i in range(0, len(batch_list)):
            pdb_name = batch_list[i]
            pdb_len = batch_list_len[i]
            notxt_flag = True
            featurefile = feature_dir + '/X-' + pdb_name + '.txt'
            if ((len(accept_list) == 1 and ('# cov' not in accept_list and '# plm' not in accept_list)) or 
                  (len(accept_list) == 2 and ('# cov' not in accept_list or '# plm' not in accept_list)) or (len(accept_list) > 2)):
                notxt_flag = False
                if not os.path.isfile(featurefile):
                    print("feature file not exists: ",featurefile, " pass!")
                    continue     
            cov = feature_dir + '/' + pdb_name + '.cov'
            if '# cov' in accept_list:
                if not os.path.isfile(cov):
                    print("Cov Matrix file not exists: ",cov, " pass!")
                    continue        
            plm = feature_dir + '/' + pdb_name + '.plm'
            if '# plm' in accept_list:
                if not os.path.isfile(plm):
                    print("plm matrix file not exists: ",plm, " pass!")
                    continue       
            targetfile = feature_dir + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
            if not os.path.isfile(targetfile):
                    print("target file not exists: ",targetfile, " pass!")
                    continue  
            (featuredata, feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, accept_list, pdb_len, notxt_flag)
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
            Y = getY(targetfile, min_seq_sep, l_max)
            if (l_max * l_max != len(Y)):
                print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                continue
            Y = Y.reshape(l_max, l_max, 1)
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

def DNCON4_1d2dconv_train_win_filter_layer_opt_fast_2D_generator(data_all_dict_padding,CV_dir,feature_dir,model_prefix,
    epoch_outside,epoch_inside,epoch_rerun,interval_len,seq_end,win_array,use_bias,hidden_type,nb_filters,nb_layers,opt,
    lib_dir, batch_size_train,path_of_lists, path_of_Y, path_of_X, Maximum_length,dist_string, reject_fea_file='None',
    initializer = "he_normal", loss_function = "categorical_crossentropy", weight_p=1.0, weight_n=1.0,  list_sep_flag=False, activation="relu"): 

    start=0
    end=seq_end
    import numpy as np
    Train_data_keys = dict()
    Train_targets_keys = dict()
    print("\n######################################\n佛祖保佑，永不迨机，永无bug，精度九十九\n######################################\n")
    feature_num=0; # the number of features for each residue
    for key in sorted(data_all_dict_padding.keys()):
        if key <start: # run first model on 100 at most
            continue
        if key > end: # run first model on 100 at most
            continue
        print('### Loading sequence length :', key)
        seq_len=key
        trainfeaturedata = data_all_dict_padding[key]
        feature_2D_num = trainfeaturedata[0][1].shape[2]
 
    print("Load feature number", feature_2D_num)
    train_avg_acc_l5_best = 0 
    val_avg_acc_l5_best = 0
    min_seq_sep = 5
    ### Define the model 
    model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
    model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
    model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)


    lr_decay = False
    train_loss_last = 1e32
    train_loss_list = []
    if model_prefix == 'DNCON4_2dCONV':
        # opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        # opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#0.001
        # opt = SGD(lr=0.01, momentum=0.9, decay=0.00, nesterov=True)
        DNCON4_CNN = DeepConv_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n, activation)
    elif model_prefix == 'DNCON4_2dINCEP':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        DNCON4_CNN = DeepInception_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dRES':
        # opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)#1
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
        # opt = SGD(lr=0.001, momentum=0.9, decay=0.00, nesterov=True)
        # opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06, decay=0.0)
        # opt = Adagrad(lr=0.01, epsilon=1e-06)
        # opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        DNCON4_CNN = DeepResnet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    elif model_prefix == 'DNCON4_2dRCNN':
        # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#0.001
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        DNCON4_CNN = DeepCovRCNN_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
    else:
        DNCON4_CNN = DeepConv_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt)

    rerun_flag=0
    # with tf.device("/cpu:0"):
    #     DNCON4_CNN = multi_gpu_model(DNCON4_CNN, gpus=2)
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    best_val_acc_out = "%s/best_validation.acc_history" % (CV_dir)
    if os.path.exists(model_weight_out):
        # print("######## Loading existing weights ",model_weight_out)
        # DNCON4_CNN.load_weights(model_weight_out)
        print("######## Loading existing weights ",model_weight_out_best)
        DNCON4_CNN.load_weights(model_weight_out_best)
        rerun_flag = 1
    else:
        print("######## Setting initial weights")   
    
        chkdirs(train_acc_history_out)     
        with open(train_acc_history_out, "a") as myfile:
          myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\n")
          
        chkdirs(val_acc_history_out)     
        with open(val_acc_history_out, "a") as myfile:
          myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\n")
        
        chkdirs(best_val_acc_out)     
        with open(best_val_acc_out, "a") as myfile:
          myfile.write("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\n")

    if loss_function == 'weighted_crossentropy':
        if weight_p < 1:
            weight_n = 1.0 - weight_p
        loss_function = _weighted_binary_crossentropy(weight_p, weight_n)
        # loss_function = _weighted_binary_crossentropy_shield(weight_p, weight_n, 5)
    else:
        loss_function = loss_function
    DNCON4_CNN.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)

    model_weight_epochs = "%s/model_weights/"%(CV_dir)
    model_predict= "%s/predict_map/"%(CV_dir)
    model_val_acc= "%s/val_acc_inepoch/"%(CV_dir)
    chkdirs(model_weight_epochs)
    chkdirs(model_predict)
    chkdirs(model_val_acc)

    tr_l = build_dataset_dictionaries_train(path_of_lists)
    te_l = build_dataset_dictionaries_test(path_of_lists)
    all_l = te_l.copy()
    train_data_num = len(tr_l)
    child_list_num = int(train_data_num/15)# 15 is the inter
    print('Total Number of Training dataset = ',str(len(tr_l)))

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    # callbacks=[reduce_lr]
    for epoch in range(epoch_rerun,epoch_outside):
        # path_of_lists, feature_dir, min_seq_sep,dist_string, batch_size, reject_fea_file='None'
        # class_weight = {0:1.,1:60.}
        if(list_sep_flag == False):
            print("\n############ Running epoch ", epoch)
            if epoch == 0 and rerun_flag == 0:
                history = DNCON4_CNN.fit_generator(generate_data_from_file(path_of_lists, feature_dir, min_seq_sep, '80', batch_size_train, reject_fea_file), steps_per_epoch = len(tr_l)//batch_size_train, epochs = 5)           
                train_loss_list.append(history.history['loss'][4])
            else:
                history = DNCON4_CNN.fit_generator(generate_data_from_file(path_of_lists, feature_dir, min_seq_sep, '80', batch_size_train, reject_fea_file), steps_per_epoch = len(tr_l)//batch_size_train, epochs = epoch_inside)  
                train_loss_list.append(history.history['loss'][0])
        else:
            for index in range(child_list_num):
                print("\n############ Runing outside epoch %i\nProcessing list %i of %i...."%(epoch, index, child_list_num))
                history = DNCON4_CNN.fit_generator(generate_data_from_file(path_of_lists, feature_dir, min_seq_sep, '80', batch_size_train, reject_fea_file, index, list_sep_flag=True), steps_per_epoch = 15//batch_size_train, epochs = 5)

        DNCON4_CNN.save_weights(model_weight_out)
        
        
        ### save models
        model_weight_out_inepoch = "%s/model-train-weight-%s-epoch%i.h5" % (model_weight_epochs,model_prefix,epoch)
        DNCON4_CNN.save_weights(model_weight_out_inepoch)
        ##### running validation

        print("Now evaluate for epoch ",epoch)
        val_acc_out_inepoch = "%s/validation_epoch%i.acc_history" % (model_val_acc, epoch) 
        sys.stdout.flush()
        print('Load all test data into memory..',end='')
        selected_list = subset_pdb_dict(te_l,   0, Maximum_length, Maximum_length, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
        print('Loading data sets ..',end='')

        testdata_len_range=50
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
        # for i in range(0, 300, testdata_len_range):
        #     p1 = {key: value for key, value in selected_list.items() if value < i + testdata_len_range and value >= i}
            print(len(p1))
            if len(p1) < 1:
                continue
            print("start predict")
            selected_list_2D = get_x_2D_from_this_list(p1, path_of_X, Maximum_length,dist_string, reject_fea_file, value)

            # cov = feature_dir + '/' + key + '.cov'
            # selected_list_2D = np.memmap(cov, dtype=np.float32, mode='r', shape=(1, 441, Maximum_length, Maximum_length))
            # selected_list_2D = selected_list_2D.transpose(0, 2, 3, 1)

            print("selected_list_2D.shape: ",selected_list_2D.shape)
            print('Loading label sets..')
            selected_list_label = get_y_from_this_list(p1, path_of_Y, 0, Maximum_length, dist_string)
            feature_2D_num = selected_list_2D.shape[3]
            DNCON4_CNN.load_weights(model_weight_out)
            DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_2D], batch_size= 1)
            ##flatteng
            # print(type(DNCON4_CNN_prediction))
            CMAP = DNCON4_CNN_prediction.reshape(Maximum_length, Maximum_length)
            Map_UpTrans = np.triu(CMAP, 1).T
            Map_UandL = np.triu(CMAP)
            real_cmap = Map_UandL + Map_UpTrans

            DNCON4_CNN_prediction = real_cmap.reshape(len(p1), Maximum_length*Maximum_length)
            (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = evaluate_prediction_4(p1, DNCON4_CNN_prediction, selected_list_label, 24)
            val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l)
            print('The best validation accuracy is ',val_acc_history_content)
            with open(val_acc_out_inepoch, "a") as myfile:
                myfile.write(val_acc_history_content)  
            out_avg_pc_l5 += avg_pc_l5 * len(p1)
            out_avg_pc_l2 += avg_pc_l2 * len(p1)
            out_avg_pc_1l += avg_pc_1l * len(p1)
            out_avg_acc_l5 += avg_acc_l5 * len(p1)
            out_avg_acc_l2 += avg_acc_l2 * len(p1)
            out_avg_acc_1l += avg_acc_1l * len(p1)
            
            step_num += 1
        print ('step_num=', step_num)
        all_num = len(selected_list)
        out_avg_pc_l5 /= all_num
        out_avg_pc_l2 /= all_num
        out_avg_pc_1l /= all_num
        out_avg_acc_l5 /= all_num
        out_avg_acc_l2 /= all_num
        out_avg_acc_1l /= all_num
        val_acc_history_content = "%i\t%i\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (interval_len,epoch,epoch_inside,out_avg_pc_l5,out_avg_pc_l2,out_avg_pc_1l,out_avg_acc_l5,out_avg_acc_l2,out_avg_acc_1l)
        with open(val_acc_history_out, "a") as myfile:
                    myfile.write(val_acc_history_content)  

        print("History of epoch %i" % epoch)
        print(history.history['loss'])
        # train_loss_list.append[history.history['loss']]
        if (lr_decay and train_loss_last != 1e32):
            current_lr = K.get_value(DNCON4_CNN.optimizer.lr)
            train_loss = history.history['loss']
            if (train_loss < train_loss_last and current_lr < 0.01):
                K.set_value(DNCON4_CNN.optimizer.lr, current_lr * 1.1)
                print("Increasing learning rate to {} ...".format(current_lr * 1.1))
            else:
                K.set_value(DNCON4_CNN.optimizer.lr, current_lr * 0.5)
                print("Decreasing learning rate to {} ...".format(current_lr * 0.5))

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

        # elif out_avg_acc_l5 < 0.1:
        #     lr = K.get_value(DNCON4_CNN.optimizer.lr)
        #     if lr >= 0.00001:
        #         K.set_value(DNCON4_CNN.optimizer.lr, lr * 0.5)
        #         print("lr changed to {}".format(lr * 0.5))
        #     DNCON4_CNN.load_weights(model_weight_out_best)
        #save predict contact map
        if epoch == epoch_outside-1:
            for key in selected_list:
                print('saving cmap of %s\n'%(key))
                value = selected_list[key]
                single_dict={key:value}
                Maximum_length = value
                # print(single_dict)
                selected_list_2D = get_x_2D_from_this_list(single_dict, path_of_X, Maximum_length,dist_string, reject_fea_file, value)
                print("selected_list_2D.shape: ",selected_list_2D.shape)
                print('Loading label sets..')
                selected_list_label = get_y_from_this_list(single_dict, path_of_Y, min_seq_sep, Maximum_length, dist_string)
                DNCON4_CNN.load_weights(model_weight_out_best)
                DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_2D], batch_size= 1)

                CMAP = DNCON4_CNN_prediction.reshape(Maximum_length, Maximum_length)
                Map_UpTrans = np.triu(CMAP, 1).T
                Map_UandL = np.triu(CMAP)
                real_cmap = Map_UandL + Map_UpTrans

                DNCON4_CNN_pred = DNCON4_CNN_prediction.reshape(len(single_dict), Maximum_length*Maximum_length)
                (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = evaluate_prediction_4(single_dict, DNCON4_CNN_pred, selected_list_label, 24)
                val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l)
                print('The best validation accuracy is ',val_acc_history_content)
                with open(best_val_acc_out, "a") as myfile:
                    myfile.write(val_acc_history_content)  
                DNCON4_CNN_prediction = DNCON4_CNN_prediction.reshape (Maximum_length, Maximum_length)
                # cmap_file = "%s/%s.txt" % (model_predict,key)
                # np.savetxt(cmap_file, DNCON4_CNN_prediction, fmt='%.4f')
                cmap_file = "%s/%s.txt" % (model_predict,key)
                np.savetxt(cmap_file, real_cmap, fmt='%.4f')

        print("Train loss history:", train_loss_list)
        #clear memory
        # K.clear_session()
        # tf.reset_default_graph()

    
    print("Training finished, best validation acc = ",val_avg_acc_l5_best)
    return val_avg_acc_l5_best