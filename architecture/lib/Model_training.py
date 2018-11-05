# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:47:26 2017

@author: Jie Hou
"""
import os

from Model_construct import *
from DNCON_lib import *


from keras.models import model_from_json,load_model, Sequential
import numpy as np
import time
import shutil
import shlex, subprocess
from subprocess import Popen, PIPE

from collections import defaultdict
#import cPickle as pickle
import pickle
# from PIL import Image

from six.moves import range

import keras.backend as K
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.constraints import maxnorm

from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D, Convolution2D
from keras.layers.normalization import BatchNormalization


import sys
import os
from shutil import copyfile
import platform
import gc

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
            else:
                DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
        
            if os.path.exists(model_weight_out):
                print("######## Loading existing weights ",model_weight_out)
                DNCON4_CNN.load_weights(model_weight_out)
                DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
            else:
                print("######## Setting initial weights")   
                DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
            #DNCON4_CNN.fit([train_1D_fea_all_array,train_2D_fea_all_array], train_targets, batch_size= batch_size_train, epochs=epoch_inside,  validation_data=([test_1D_fea_all_array,test_2D_fea_all_array], test_label_all_array), verbose=1)
            if key < 150 and epoch ==0:
              DNCON4_CNN.fit([train_1D_fea_all_array,train_2D_fea_all_array], train_targets, batch_size= batch_size_train_new, epochs=40, verbose=1)
            else:
              DNCON4_CNN.fit([train_1D_fea_all_array,train_2D_fea_all_array], train_targets, batch_size= batch_size_train_new, epochs=epoch_inside, verbose=1)
            DNCON4_CNN.save_weights(model_weight_out)
            # serialize model to JSON
            # model_json = DNCON4_CNN.to_json()
            # print("Saved model to disk")
            # with open(model_out, "w") as json_file:
            #     json_file.write(model_json)
            # del train_1D_fea_select
            # del train_2D_fea_select
            # del train_featuredata_all
            # del train_1D_fea_all_array
            # del train_2D_fea_all_array
            # del train_1D_fea_all_array_expand
            # del train_2D_fea_all_array_expand
            # del train_label_all_array_expand
            # del train_targets
            # del label_select
            gc.collect()
            # del test_featuredata_all 
            # del test_targets
            
        
        ### save models
        model_weight_out_inepoch = "%s/model-train-weight-%s-epoch%i.h5" % (CV_dir,model_prefix,epoch)
        DNCON4_CNN.save_weights(model_weight_out_inepoch)
        weight_array= DNCON4_CNN.get_weights()
        weight_filename="%s/model-train-weight-%s-epoch%i.txt" % (CV_dir,model_prefix,epoch)
        fileObject = open(weight_filename, 'w')
        for i in range(len(weight_array)):
            fileObject.write(str(weight_array[i])+'\n')
        fileObject.close()

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
        else:
            DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num_vali,feature_2D_num_vali,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)

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



def DNCON4_1d2dconv_train_win_filter_layer_opt_fast_2Donly(data_all_dict_padding,testdata_all_dict_padding,CV_dir,feature_dir,model_prefix,epoch_outside,epoch_inside,interval_len,seq_end,win_array,use_bias,hidden_type,nb_filters,nb_layers,opt,lib_dir, batch_size_train,path_of_lists,path_of_Y, path_of_X,Maximum_length,dist_string): 
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
          feature_2D_num=train_2D_feature.shape[2]  
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
            model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
            model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
            model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)
            
            sequence_length = seq_len
            
            print("######## Setting initial model based on length ",sequence_length)
            ## ktop_node is the length of input proteins
            if model_prefix == 'DNCON4_1d2dconv':
                DNCON4_CNN = DNCON4_with_paras_2D(win_array,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
            else:
                DNCON4_CNN = DNCON4_with_paras_2D(win_array,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
        
            if os.path.exists(model_weight_out):
                print("######## Loading existing weights ",model_weight_out)
                DNCON4_CNN.load_weights(model_weight_out)
                DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
            else:
                print("######## Setting initial weights")
                DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
            
            
            #DNCON4_CNN.fit([train_1D_fea_all_array,train_2D_fea_all_array], train_targets, batch_size= batch_size_train, epochs=epoch_inside,  validation_data=([test_1D_fea_all_array,test_2D_fea_all_array], test_label_all_array), verbose=1)
            DNCON4_CNN.fit([train_2D_fea_all_array], train_targets, batch_size= batch_size_train_new, epochs=epoch_inside, verbose=1)
            DNCON4_CNN.save_weights(model_weight_out)
            # serialize model to JSON
            # model_json = DNCON4_CNN.to_json()
            # print("Saved model to disk")
            # with open(model_out, "w") as json_file:
            #     json_file.write(model_json)

            del train_featuredata_all
            del train_targets
            del test_featuredata_all
            del test_targets
        
        
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
        selected_list_2D = get_x_2D_from_this_list(selected_list, path_of_X, Maximum_length,dist_string)
        print("selected_list_2D.shape: ",selected_list_2D.shape)
        print('Loading label sets..')
        selected_list_label = get_y_from_this_list(selected_list, path_of_Y, 24, Maximum_length, dist_string)
        feature_2D_num = selected_list_2D.shape[3]
        DNCON4_CNN = DNCON4_with_paras_2D(win_array,feature_2D_num,Maximum_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
        DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
        DNCON4_CNN.load_weights(model_weight_out)
        DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_2D], batch_size= 1)
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
        
    
    print("Training finished, best validation acc = ",val_avg_acc_l5_best)
    # print "Training finished, best training acc = ",train_acc_best
    print("Setting and saving best weights")
    DNCON4_CNN.load_weights(model_weight_out_best)
    DNCON4_CNN.save_weights(model_weight_out)

