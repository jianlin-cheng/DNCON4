##(1) cp -ar /storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/CNN_arch/  /scratch/jh7x3/DNCON4/architecture/
##(2) cd /scratch/jh7x3/DNCON4/architecture/CNN_arch/scripts
##(3) load keras1 source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/Keras1.2_TF1.5/bin/activate
##(4) python


import sys
import os
from shutil import copyfile
import platform

import os
import numpy as np
import math
import os
import sys
import random
import keras.backend as K
epsilon = K.epsilon()

GLOBAL_PATH= '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/CNN_arch/'
print GLOBAL_PATH
sys.path.insert(0, GLOBAL_PATH+'/lib/')


############# (1) test feature loading

from Data_loading import getX_1D_2D,load_train_test_data_padding_with_interval


featurefile = '/scratch/jh7x3/DNCON4/data/badri_training_benchmark/feats/X-1B9O-A.txt'
featuredata = getX_1D_2D(featurefile, reject_fea_file='None')
featuredata.keys()

feature_1D_all=[]
feature_2D_all=[]
for key in featuredata.keys():
    myarray = np.asarray(featuredata[key])
    featuredata[key] = myarray.reshape(len(myarray),myarray.shape[1],myarray.shape[2])
    print "keys: ", key, ": ", featuredata[key].shape 
    if featuredata[key].shape[1] == featuredata[key].shape[2]:
      feature_2D_all.append(featuredata[key])
    else:
      feature_1D_all.append(featuredata[key])

for i in range(0,len(feature_1D_all)):
    print i, ": ", feature_1D_all[i].shape


feature_1D_all_tmp = np.concatenate(feature_1D_all)
feature_2D_all_tmp = np.concatenate(feature_2D_all)
feature_1D_all_tmp.shape  # (22, 1, 123)
feature_2D_all_tmp.shape # (18, 123, 123)
feature_1D_all_complete =  feature_1D_all_tmp.reshape(feature_1D_all_tmp.shape[0],feature_1D_all_tmp.shape[2]).transpose()
feature_1D_all_complete.shape #(123, 22)
feature_2D_all_complete =  feature_2D_all_tmp.reshape(feature_2D_all_tmp.shape[1],feature_2D_all_tmp.shape[2],feature_2D_all_tmp.shape[0])
feature_2D_all_complete.shape #(123, 123, 18)


############# test data loading

data_list = '/scratch/jh7x3/DNCON4/data/badri_training_benchmark/lists-test-train/test.lst'
feature_dir = '/scratch/jh7x3/DNCON4/data/badri_training_benchmark/feats/'
Interval=20
seq_end=1000
data_all_dict_padding = load_train_test_data_padding_with_interval(data_list, feature_dir,Interval,seq_end,24,10)



############# test network construction 

from Model_construct import DNCON4_with_paras

win_array = [6]
feature_1D_num=25
feature_2D_num=18
sequence_length=50
use_bias=True
hidden_type='sigmoid'
nb_filters=5
nb_layers=6
opt='nadam'
batch_size_train=25
epoch_inside = 3 

DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt)


key=60
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


testfeaturedata = trainfeaturedata

test_label_all = []
test_1D_fea_all = []
test_2D_fea_all = []
for i in range(0,len(testfeaturedata)):
  test_labels = testfeaturedata[i][0] ## (seq_len*seq_len,)
  test_1D_feature = testfeaturedata[i][1] ## (seq_len, 1d_fea_num)
  test_2D_feature = testfeaturedata[i][2] ## (seq_len, seq_len, 2d_fea_num)
  test_label_all.append(test_labels)
  test_1D_fea_all.append(test_1D_feature)
  test_2D_fea_all.append(test_2D_feature)


train_label_all_array = np.asarray(train_label_all) #(21, 48400)
train_1D_fea_all_array = np.asarray(train_1D_fea_all) #(21, 220, 22)
train_2D_fea_all_array = np.asarray(train_2D_fea_all) #(21, 220, 220, 18)  
train_2D_fea_all_array = train_2D_fea_all_array.reshape(train_2D_fea_all_array.shape[0],train_2D_fea_all_array.shape[1]*train_2D_fea_all_array.shape[2],train_2D_fea_all_array.shape[3])     #(21, 220*220, 18)  


test_label_all_array = np.asarray(test_label_all) #(21, 48400)
test_1D_fea_all_array = np.asarray(test_1D_fea_all) #(21, 220, 22)
test_2D_fea_all_array = np.asarray(test_2D_fea_all) #(21, 220, 220, 18)        
test_2D_fea_all_array = test_2D_fea_all_array.reshape(test_2D_fea_all_array.shape[0],test_2D_fea_all_array.shape[1]*test_2D_fea_all_array.shape[2],test_2D_fea_all_array.shape[3])  #(21, 220*220, 18)  

epoch = 1 

print "Train 1D shape: ",train_1D_fea_all_array.shape, " in outside epoch ", epoch 
print "Train 2D shape: ",train_2D_fea_all_array.shape, " in outside epoch ", epoch 
print "Test 1D shape: ",test_1D_fea_all_array.shape, " in outside epoch ", epoch
print "Test 2D shape: ",test_2D_fea_all_array.shape, " in outside epoch ", epoch


### Define the model 

sequence_length = seq_len

print "######## Setting initial model based on length ",sequence_length;
DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt)

DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)


DNCON4_CNN.fit([train_1D_fea_all_array,train_2D_fea_all_array], train_label_all_array, batch_size= batch_size_train, nb_epoch=epoch_inside,  validation_data=([test_1D_fea_all_array,test_2D_fea_all_array], test_label_all_array), verbose=1)

DNCON4_CNN_prediction = DNCON4_CNN.predict([train_1D_fea_all_array,train_2D_fea_all_array])

DNCON4_CNN_prediction.shape



