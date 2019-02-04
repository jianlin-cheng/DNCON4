##(1) cp -ar /storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/CNN_arch/  /scratch/zggc9/DNCON4/architecture/
##(2) cd /scratch/zggc9/DNCON4/architecture/CNN_arch/scripts
## load keras2 
##(3) source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate
##(4) python


import sys
import os
from shutil import copyfile
import platform
import numpy as np
import math
import random
import keras.backend as K

#%%
epsilon = K.epsilon()
current_os_name = platform.platform()
print (('%s' )% current_os_name)
if current_os_name == 'Linux-4.15.0-36-generic-x86_64-with-Ubuntu-18.04-bionic': #on local
  # GLOBAL_PATH='/mnt/data/zhiye/Python/DNCON4'
  GLOBAL_PATH= '/mnt/data/zhiye/Python/DNCON4/architecture/'
  featurefile = '/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/feats/X-1B9O-A.txt'
  data_list = '/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/lists-test-train/test.lst'
  feature_dir = '/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/feats/'
  CV_dir = '/mnt/data/zhiye/Python/DNCON4/architecture/CNN_arch/test'
elif current_os_name == 'Linux-3.10.0-862.14.4.el7.x86_64-x86_64-with-centos-7.5.1804-Core': #on lewis
  # GLOBAL_PATH=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
  GLOBAL_PATH= '/scratch/jh7x3/DNCON4/architecture/'
  featurefile = '/scratch/jh7x3/DNCON4/data/badri_training_benchmark/feats/X-1B9O-A.txt'
  data_list = '/scratch/jh7x3/DNCON4/data/badri_training_benchmark/lists-test-train_sample20/test.lst'
  feature_dir = '/scratch/jh7x3/DNCON4/data/badri_training_benchmark/feats/'
  CV_dir = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/CNN_arch/test'
else:
  print ('Please check current operate system!')
  sys.exit(1)

print(GLOBAL_PATH)
sys.path.insert(0, GLOBAL_PATH+'/lib/')
#%%

############# (1) test feature loading

from Data_loading import getX_1D_2D,load_train_test_data_padding_with_interval


#%%
    
### load all features into 1D and 2D format
(featuredata,feature_index_all_dict) = getX_1D_2D(featurefile, reject_fea_file='None')
featuredata.keys()

feature_1D_all=[]
feature_2D_all=[]
for key in sorted(featuredata.keys()):
    featurename = feature_index_all_dict[key]
    feature = featuredata[key]
    feature = np.asarray(feature)
    print("keys: ", key, " featurename: ",featurename, " feature_shape:", feature.shape)
    #print "keys: ", key, ": ", featuredata[key].shape
    
    if feature.shape[0] == feature.shape[1]:
      feature_2D_all.append(feature)
    else:
      feature_1D_all.append(feature)

fea_len = feature_2D_all[0].shape[0]
F_2D = len(feature_2D_all)

X_2D = np.zeros((fea_len, fea_len, F_2D))
for m in range (0, F_2D):
  X_2D[0:fea_len, 0:fea_len, m] = feature_2D_all[m]
  

F_1D = len(feature_1D_all)

X_1D = np.zeros((fea_len, F_1D))
for m in range (0, F_1D):
  X_1D[0:fea_len, m] = feature_1D_all[m]


feature_1D_all_complete =  X_1D
feature_1D_all_complete.shape #(123, 22)
feature_2D_all_complete =  X_2D
feature_2D_all_complete.shape #(123, 123, 18)


############# test data loading

Interval=20
seq_end=1000
data_all_dict_padding = load_train_test_data_padding_with_interval(data_list, feature_dir,Interval,seq_end,24,10)



############# test network construction 
#%%
from Model_construct import DNCON4_with_paras,DeepCovRCNN_with_paras,DeepResnet_with_paras,DeepInception_with_paras,DeepCovResAtt_with_paras
from Model_construct import DeepCRMN_with_paras, DeepFracNet_with_paras
win_array = [6]
feature_1D_num=25
feature_2D_num=18
sequence_length=100
use_bias=True
hidden_type='sigmoid'
nb_filters=5
nb_layers=2
opt='nadam'
batch_size_train=5
epoch_inside = 3 

#DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train)
#DNCON4_CNN = DeepCRMN_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train)
#DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
#DNCON4_CNN.save_weights('50.h5')
#DNCON4_CNN.load_weights('50.h5')
#DNCON4_CNN.summary()
#%%
key=80
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
#train_2D_fea_all_array = train_2D_fea_all_array.reshape(train_2D_fea_all_array.shape[0],train_2D_fea_all_array.shape[1]*train_2D_fea_all_array.shape[2],train_2D_fea_all_array.shape[3])     #(21, 220*220, 18)  


test_label_all_array = np.asarray(test_label_all) #(21, 48400)
test_1D_fea_all_array = np.asarray(test_1D_fea_all) #(21, 220, 22)
test_2D_fea_all_array = np.asarray(test_2D_fea_all) #(21, 220, 220, 18)        
#test_2D_fea_all_array = test_2D_fea_all_array.reshape(test_2D_fea_all_array.shape[0],test_2D_fea_all_array.shape[1]*test_2D_fea_all_array.shape[2],test_2D_fea_all_array.shape[3])  #(21, 220*220, 18)  

epoch = 1 

print("Train 1D shape: ",train_1D_fea_all_array.shape, " in outside epoch ", epoch)
print("Train 2D shape: ",train_2D_fea_all_array.shape, " in outside epoch ", epoch)
print("Test 1D shape: ",test_1D_fea_all_array.shape, " in outside epoch ", epoch)
print("Test 2D shape: ",test_2D_fea_all_array.shape, " in outside epoch ", epoch)


### Define the model 

sequence_length = seq_len

print("######## Setting initial model based on length ",sequence_length)

#### expand dataset to batchsize 
batch_size_train=5


##### contruct another dataset large datase for test from sample 2 to sample 7, 

factor = train_label_all_array.shape[0] % batch_size_train  # 15 % 2 = 1
num_to_expand = 5
random_to_pick = np.random.randint(0,train_label_all_array.shape[0],num_to_expand)
train_1D_fea_all_array_new = np.zeros((train_label_all_array.shape[0]+num_to_expand,train_1D_fea_all_array.shape[1],train_1D_fea_all_array.shape[2]))
train_1D_fea_all_array_new[:train_1D_fea_all_array.shape[0],:train_1D_fea_all_array.shape[1],:train_1D_fea_all_array.shape[2]] = train_1D_fea_all_array

train_2D_fea_all_array_new = np.zeros((train_label_all_array.shape[0]+num_to_expand,train_2D_fea_all_array.shape[1],train_2D_fea_all_array.shape[2],train_2D_fea_all_array.shape[3]))
train_2D_fea_all_array_new[:train_2D_fea_all_array.shape[0],:train_2D_fea_all_array.shape[1],:train_2D_fea_all_array.shape[2],:train_2D_fea_all_array.shape[3]] = train_2D_fea_all_array

train_label_all_array_new = np.zeros((train_label_all_array.shape[0]+num_to_expand,train_label_all_array.shape[1]))
train_label_all_array_new[:train_label_all_array.shape[0],:train_label_all_array.shape[1]] = train_label_all_array



for indx in range(0,len(random_to_pick)):
  label_select = train_label_all_array[random_to_pick[indx],:]
  train_1D_fea_select = train_1D_fea_all_array[random_to_pick[indx],:,:]
  train_2D_fea_select = train_2D_fea_all_array[random_to_pick[indx],:,:]
  
  train_label_all_array_new[train_label_all_array.shape[0]+indx,:] = label_select
  train_1D_fea_all_array_new[train_1D_fea_all_array.shape[0]+indx,:,:] = train_1D_fea_select
  train_2D_fea_all_array_new[train_2D_fea_all_array.shape[0]+indx,:,:,:] = train_2D_fea_select


print("The train label size: ",train_label_all_array_new.shape)
print("The train label size: ",train_1D_fea_all_array_new.shape)
print("The train label size: ",train_2D_fea_all_array_new.shape)


######## expand training dataset for batch training
batch_size_train_new = batch_size_train
if train_label_all_array_new.shape[0] < batch_size_train:
  print("Setting batch size from ",batch_size_train, " to ",train_label_all_array_new.shape[0])
  batch_size_train_new = train_label_all_array_new.shape[0]
else:
  factor = int(train_label_all_array_new.shape[0] / batch_size_train)  # 7/5 = 1
  num_to_expand = (factor+1)*batch_size_train - train_label_all_array_new.shape[0]
  random_to_pick = np.random.randint(0,train_label_all_array_new.shape[0],num_to_expand)
  train_1D_fea_all_array_expand = np.zeros((train_label_all_array_new.shape[0]+num_to_expand,train_1D_fea_all_array_new.shape[1],train_1D_fea_all_array_new.shape[2]))
  train_1D_fea_all_array_expand[:train_1D_fea_all_array_new.shape[0],:train_1D_fea_all_array_new.shape[1],:train_1D_fea_all_array_new.shape[2]] = train_1D_fea_all_array_new
  
  train_2D_fea_all_array_expand = np.zeros((train_label_all_array_new.shape[0]+num_to_expand,train_2D_fea_all_array_new.shape[1],train_2D_fea_all_array_new.shape[2],train_2D_fea_all_array_new.shape[3]))
  train_2D_fea_all_array_expand[:train_2D_fea_all_array_new.shape[0],:train_2D_fea_all_array_new.shape[1],:train_2D_fea_all_array_new.shape[2],:train_2D_fea_all_array_new.shape[3]] = train_2D_fea_all_array_new
  
  train_label_all_array_expand = np.zeros((train_label_all_array_new.shape[0]+num_to_expand,train_label_all_array_new.shape[1]))
  train_label_all_array_expand[:train_label_all_array_new.shape[0],:train_label_all_array_new.shape[1]] = train_label_all_array_new
  for indx in range(0,len(random_to_pick)):
    label_select = train_label_all_array_new[random_to_pick[indx],:]
    train_1D_fea_select = train_1D_fea_all_array_new[random_to_pick[indx],:,:]
    train_2D_fea_select = train_2D_fea_all_array_new[random_to_pick[indx],:,:]
    
    train_label_all_array_expand[train_label_all_array_new.shape[0]+indx,:] = label_select
    train_1D_fea_all_array_expand[train_1D_fea_all_array_new.shape[0]+indx,:,:] = train_1D_fea_select
    train_2D_fea_all_array_expand[train_2D_fea_all_array_new.shape[0]+indx,:,:,:] = train_2D_fea_select
  
  train_1D_fea_all_array_new = train_1D_fea_all_array_expand
  train_2D_fea_all_array_new = train_2D_fea_all_array_expand
  train_label_all_array_new = train_label_all_array_expand

print("The expanded train label size: ",train_label_all_array_new.shape)
print("The expanded train 1D fea size: ",train_1D_fea_all_array_new.shape)
print("The expanded train 2D fea size: ",train_2D_fea_all_array_new.shape)

#%%
# DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
# DNCON4_CNN = DeepCovRCNN_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
# DNCON4_CNN = DeepResnet_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
# DNCON4_CNN = DeepInception_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
# DNCON4_CNN = DeepCovResAtt_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
DNCON4_CNN = DeepCRMN_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
# DNCON4_CNN = DeepFracNet_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)


DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
DNCON4_CNN.fit([train_1D_fea_all_array_new,train_2D_fea_all_array_new], train_label_all_array_new, batch_size= batch_size_train_new, epochs=epoch_inside, verbose=1)

model_prefix = 'conv2d'
model_out= "%s/model-train-%s-%i.json" % (CV_dir, model_prefix, seq_len)
model_weight_out = "%s/model-train-weight-%s-%i.h5" % (CV_dir, model_prefix, seq_len)
model_weight_out_best = "%s/model-train-weight-%s-%i-best-val.h5" % (CV_dir, model_prefix, seq_len)

model_json = DNCON4_CNN.to_json()
print("Saved model to disk")
model_out
with open(model_out, "w") as json_file:
    json_file.write(model_json)

print("Saved best weight to disk, ")
DNCON4_CNN.save_weights(model_weight_out_best)
#DNCON4_CNN.fit([train_1D_fea_all_array,train_2D_fea_all_array], train_label_all_array, batch_size= batch_size_train, nb_epoch=epoch_inside,  validation_data=([test_1D_fea_all_array,test_2D_fea_all_array], test_label_all_array), verbose=1)

#%%
# DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
# DNCON4_CNN = DeepCovRCNN_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
# DNCON4_CNN = DeepResnet_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
# DNCON4_CNN = DeepInception_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
# DNCON4_CNN = DeepCovResAtt_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
DNCON4_CNN = DeepCRMN_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
# DNCON4_CNN = DeepFracNet_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)

DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
#DNCON4_CNN.load_weights(model_weight_out_best)
DNCON4_CNN.load_weights( "%s/model-train-weight-%s-%i-best-val.h5" % (CV_dir, model_prefix, seq_len))
DNCON4_CNN_prediction = DNCON4_CNN.predict([train_1D_fea_all_array_new,train_2D_fea_all_array_new], batch_size= 1)

DNCON4_CNN_prediction.shape

# #%%
# from keras.models import model_from_json
# jsonfile="%s/model-train-%s-%i.json" % (CV_dir, model_prefix, 300)
# json_file_model = open(jsonfile, 'r')
# loaded_model_json = json_file_model.read()
# json_file_model.close()
# weightfile="%s/model-train-weight-%s-%i-best-val.h5" % (CV_dir, model_prefix, 300)
# DNCON4_CNN_300 = model_from_json(loaded_model_json) 
# DNCON4_CNN_300.load_weights(weightfile)
# #%%
# weight60= DNCON4_CNN.get_weights()
# fileObject = open('weight60.txt', 'w')
# for i in range(len(weight60)):
#     fileObject.write(str(weight60[i])+'\n')
# fileObject.close()
# #%%
# weight300= DNCON4_CNN.get_weights()
# fileObject = open('weight300.txt', 'w')
# for i in range(len(weight300)):
#     fileObject.write(str(weight300[i])+'\n')
# fileObject.close()
