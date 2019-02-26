##(1) cp -ar /storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/CNN_arch/  /scratch/zggc9/DNCON4/architecture/
##(2) cd /scratch/zggc9/DNCON4/architecture/CNN_arch/scripts
## load keras2 
##(3) source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate
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

#GLOBAL_PATH= '/scratch/zggc9/DNCON4/architecture/CNN_arch/'
GLOBAL_PATH= '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/CNN_arch/'
print(GLOBAL_PATH)
sys.path.insert(0, GLOBAL_PATH+'/lib/')


############# (1) test feature loading

from Data_loading import getX_1D_2D,load_train_test_data_padding_with_interval


featurefile = '/scratch/zggc9/DNCON4/data/badri_training_benchmark/feats/X-1B9O-A.txt'
featuredata = getX_1D_2D(featurefile, reject_fea_file='None')
featuredata.keys()

feature_1D_all=[]
feature_2D_all=[]
for key in featuredata.keys():
    myarray = np.asarray(featuredata[key])
    featuredata[key] = myarray.reshape(len(myarray),myarray.shape[1],myarray.shape[2])
    print("keys: ", key, ": ", featuredata[key].shape)
    if featuredata[key].shape[1] == featuredata[key].shape[2]:
      feature_2D_all.append(featuredata[key])
    else:
      feature_1D_all.append(featuredata[key])

for i in range(0,len(feature_1D_all)):
    print(i, ": ", feature_1D_all[i].shape)


feature_1D_all_tmp = np.concatenate(feature_1D_all)
feature_2D_all_tmp = np.concatenate(feature_2D_all)
feature_1D_all_tmp.shape  # (22, 1, 123)
feature_2D_all_tmp.shape # (18, 123, 123)
feature_1D_all_complete =  feature_1D_all_tmp.reshape(feature_1D_all_tmp.shape[0],feature_1D_all_tmp.shape[2]).transpose()
feature_1D_all_complete.shape #(123, 22)
feature_2D_all_complete =  feature_2D_all_tmp.reshape(feature_2D_all_tmp.shape[1],feature_2D_all_tmp.shape[2],feature_2D_all_tmp.shape[0])
feature_2D_all_complete.shape #(123, 123, 18)


############# test data loading

data_list = '/scratch/zggc9/DNCON4/data/badri_training_benchmark/lists-test-train/test.lst'
feature_dir = '/scratch/zggc9/DNCON4/data/badri_training_benchmark/feats/'
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
batch_size_train=5
epoch_inside = 3 

DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train)


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



DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
DNCON4_CNN.fit([train_1D_fea_all_array_new,train_2D_fea_all_array_new], train_label_all_array_new, batch_size= batch_size_train_new, nb_epoch=epoch_inside, verbose=1)

CV_dir = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/CNN_arch/test'
model_prefix = 'conv2d'
model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)

model_json = DNCON4_CNN.to_json()
print("Saved model to disk")
model_out
with open(model_out, "w") as json_file:
    json_file.write(model_json)

print("Saved best weight to disk, ")
DNCON4_CNN.save_weights(model_weight_out_best)
#DNCON4_CNN.fit([train_1D_fea_all_array,train_2D_fea_all_array], train_label_all_array, batch_size= batch_size_train, nb_epoch=epoch_inside,  validation_data=([test_1D_fea_all_array,test_2D_fea_all_array], test_label_all_array), verbose=1)

DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
DNCON4_CNN.load_weights(model_weight_out_best)
DNCON4_CNN_prediction = DNCON4_CNN.predict([train_1D_fea_all_array_new,train_2D_fea_all_array_new], batch_size= 1)

DNCON4_CNN_prediction.shape



################### Evaluate the prediction

def build_dataset_dictionaries(path_lists):
  length_dict = {}
  n_dict = {}
  neff_dict = {}
  with open(path_lists + 'L.txt') as f:
    for line in f:
      cols = line.strip().split()
      length_dict[cols[0]] = int(cols[1])
  with open(path_lists + 'N.txt') as f:
    for line in f:
      cols = line.strip().split()
      n_dict[cols[0]] = int(float(cols[1]))
  with open(path_lists + 'Neff.txt') as f:
    for line in f:
      cols = line.strip().split()
      neff_dict[cols[0]] = int(float(cols[1]))
  tr_l = {}
  tr_n = {}
  tr_e = {}
  with open(path_lists + 'train.lst') as f:
    for line in f:
      tr_l[line.strip()] = length_dict[line.strip()]
      tr_n[line.strip()] = n_dict[line.strip()]
      tr_e[line.strip()] = neff_dict[line.strip()]
  te_l = {}
  te_n = {}
  te_e = {}
  with open(path_lists + 'test.lst') as f:
    for line in f:
      te_l[line.strip()] = length_dict[line.strip()]
      te_n[line.strip()] = n_dict[line.strip()]
      te_e[line.strip()] = neff_dict[line.strip()]
  print ('')
  print ('Data counts:')
  print ('Total : ' + str(len(length_dict)))
  print ('Train : ' + str(len(tr_l)))
  print ('Test  : ' + str(len(te_l)))
  print ('')
  return (tr_l, tr_n, tr_e, te_l, te_n, te_e)

def subset_pdb_dict(dict, minL, maxL, count, randomize_flag):
  selected = {}
  # return a dict with random 'X' PDBs
  if (randomize_flag == 'random'):
    pdbs = dict.keys()
    random.shuffle(pdbs)
    i = 0
    for pdb in pdbs:
      if (dict[pdb] > minL and dict[pdb] <= maxL):
        selected[pdb] = dict[pdb]
        i = i + 1
        if i == count:
          break
  # return first 'X' PDBs sorted by L
  if (randomize_flag == 'ordered'):
    i = 0
    for key, value in sorted(dict.items(), key=lambda  x: x[1]):
      if (dict[key] > minL and dict[key] <= maxL):
        selected[key] = value
        i = i + 1
        if i == count:
          break
  return selected

def get_y_from_this_list(selected_ids, path, min_seq_sep, l_max, y_dist):
  xcount = len(selected_ids)
  sample_pdb = ''
  for pdb in selected_ids:
    sample_pdb = pdb
    break
  y = getY(path + 'Y' + y_dist + '-' + sample_pdb + '.txt', min_seq_sep, l_max)
  if (l_max * l_max != len(y)):
    print ('Error!! y does not have L * L feature values!!')
    sys.exit()
  Y = np.zeros((xcount, l_max * l_max))
  i = 0
  for pdb in sorted(selected_ids):
    Y[i, :]       = getY(path + 'Y' + y_dist + '-' + pdb + '.txt', min_seq_sep, l_max)
    i = i + 1
  return Y

def getY(true_file, min_seq_sep, l_max):
  # calcualte the length of the protein (the first feature)
  L = 0
  with open(true_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      L = line.strip().split()
      L = len(L)
      break
  Y = np.zeros((l_max, l_max))
  i = 0
  with open(true_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      this_line = line.strip().split()
      Y[i, 0:L] = feature2D = np.asarray(this_line)
      i = i + 1
  for p in range(0,L):
    for q in range(0,L):
      # updated only for the last project 'p19' to test the effect
      if ( abs(q - p) < min_seq_sep):
        Y[p][q] = 0
  Y = Y.flatten()
  return Y


def get_x_from_this_list(selected_ids, path, l_max):
  xcount = len(selected_ids)
  sample_pdb = ''
  for pdb in selected_ids:
    sample_pdb = pdb
    break
  print(path,'/X-',sample_pdb,'.txt')
  x = getX(path + 'X-'  + sample_pdb + '.txt', l_max)
  F = len(x[0, 0, :])
  X = np.zeros((xcount, l_max, l_max, F))
  i = 0
  for pdb in sorted(selected_ids):
    T = getX(path + 'X-'  + pdb + '.txt', l_max)
    if len(T[0, 0, :]) != F:
      print('ERROR! Feature length of ',sample_pdb,' not equal to ',pdb)
    X[i, :, :, :] = T
    i = i + 1
  return X




def get_x_1D_2D_from_this_list(selected_ids, feature_dir, l_max):
  xcount = len(selected_ids)
  sample_pdb = ''
  for pdb in selected_ids:
    sample_pdb = pdb
    break
  featurefile =feature_dir + 'X-'  + sample_pdb + '.txt'
  print(featurefile)
  ### load the data
  featuredata = getX_1D_2D(featurefile, reject_fea_file='None')     
  ### merge 1D data to L*m
  ### merge 2D data to  L*L*n
  feature_1D_all=[]
  feature_2D_all=[]
  for key in featuredata.keys():
      myarray = np.asarray(featuredata[key])
      featuredata[key] = myarray.reshape(len(myarray),myarray.shape[1],myarray.shape[2])
      #print "keys: ", key, ": ", featuredata[key].shape
      if featuredata[key].shape[1] == featuredata[key].shape[2]:
        feature_2D_all.append(featuredata[key])
      else:
        feature_1D_all.append(featuredata[key])
  #for i in range(0,len(feature_1D_all)):
  #    print i, ": ", feature_1D_all[i].shape      
  feature_1D_all_tmp = np.concatenate(feature_1D_all)
  feature_2D_all_tmp = np.concatenate(feature_2D_all)
  #print feature_1D_all_tmp.shape  # (22, 1, 123)
  #print feature_2D_all_tmp.shape # (18, 123, 123)   
  feature_1D_all_complete =  feature_1D_all_tmp.reshape(feature_1D_all_tmp.shape[0],feature_1D_all_tmp.shape[2]).transpose()
  #print feature_1D_all_complete.shape #(123, 22)
  feature_2D_all_complete =  feature_2D_all_tmp.reshape(feature_2D_all_tmp.shape[1],feature_2D_all_tmp.shape[2],feature_2D_all_tmp.shape[0])
  #print feature_2D_all_complete.shape #(123, 123, 18)  
  fea_len = feature_2D_all_complete.shape[0]
  F_1D = len(feature_1D_all_complete[0, :])
  F_2D = len(feature_2D_all_complete[0, 0, :])
  X_1D = np.zeros((xcount, l_max, F_1D))
  X_2D = np.zeros((xcount, l_max, l_max, F_2D))
  i = 0
  for pdb_name in sorted(selected_ids):
      print(pdb_name, "..",end='')
      featurefile = feature_dir + '/X-' + pdb_name + '.txt'
      if not os.path.isfile(featurefile):
                  print("feature file not exists: ",featurefile, " pass!")
                  continue         
      targetfile = feature_dir + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
      if not os.path.isfile(targetfile):
                  print("target file not exists: ",targetfile, " pass!")
                  continue
      ### load the data
      featuredata = getX_1D_2D(featurefile, reject_fea_file='None')     
      ### merge 1D data to L*m
      ### merge 2D data to  L*L*n
      feature_1D_all=[]
      feature_2D_all=[]
      for key in featuredata.keys():
          myarray = np.asarray(featuredata[key])
          featuredata[key] = myarray.reshape(len(myarray),myarray.shape[1],myarray.shape[2])
          #print "keys: ", key, ": ", featuredata[key].shape
          if featuredata[key].shape[1] == featuredata[key].shape[2]:
            feature_2D_all.append(featuredata[key])
          else:
            feature_1D_all.append(featuredata[key])
      
      #for i in range(0,len(feature_1D_all)):
      #    print i, ": ", feature_1D_all[i].shape      
      feature_1D_all_tmp = np.concatenate(feature_1D_all)
      feature_2D_all_tmp = np.concatenate(feature_2D_all)
      #print feature_1D_all_tmp.shape  # (22, 1, 123)
      #print feature_2D_all_tmp.shape # (18, 123, 123)   
      feature_1D_all_complete =  feature_1D_all_tmp.reshape(feature_1D_all_tmp.shape[0],feature_1D_all_tmp.shape[2]).transpose()
      #print feature_1D_all_complete.shape #(123, 22)
      feature_2D_all_complete =  feature_2D_all_tmp.reshape(feature_2D_all_tmp.shape[1],feature_2D_all_tmp.shape[2],feature_2D_all_tmp.shape[0])
      #print feature_2D_all_complete.shape #(123, 123, 18) 
      if len(feature_1D_all_complete[0, :]) != F_1D:
        print('ERROR! 1D Feature length of ',sample_pdb,' not equal to ',pdb_name)
        X_1D[i, :, :] = feature_1D_all_complete
      if len(feature_2D_all_complete[0, 0, :]) != F_2D:
        print('ERROR! 2D Feature length of ',sample_pdb,' not equal to ',pdb_name)
        X_2D[i, :, :, :] = feature_2D_all_complete
      i = i + 1
  return (X_1D,X_2D)

def evaluate_prediction (dict_l, dict_n, dict_e, P, Y, min_seq_sep):
  P2 = floor_lower_left_to_zero(P, min_seq_sep)
  datacount = len(Y[:, 0])
  L = int(math.sqrt(len(Y[0, :])))
  Y1 = floor_lower_left_to_zero(Y, min_seq_sep)
  list_acc_l5 = []
  list_acc_l2 = []
  list_acc_1l = []
  P3L5 = ceil_top_xL_to_one(dict_l, P2, Y, 0.2)
  P3L2 = ceil_top_xL_to_one(dict_l, P2, Y, 0.5)
  P31L = ceil_top_xL_to_one(dict_l, P2, Y, 1)
  (list_acc_l5, list_acc_l2, list_acc_1l) = print_detailed_evaluations(dict_l, dict_n, dict_e, P3L5, P3L2, P31L, Y)
  return (list_acc_l5, list_acc_l2, list_acc_1l)

# Floor everything below the triangle of interest to zero
def floor_lower_left_to_zero(XP, min_seq_sep):
  X = np.copy(XP)
  datacount = len(X[:, 0])
  L = int(math.sqrt(len(X[0, :])))
  X_reshaped = X.reshape(datacount, L, L)
  for p in range(0,L):
    for q in range(0,L):
      if ( q - p < min_seq_sep):
        X_reshaped[:, p, q] = 0
  X = X_reshaped.reshape(datacount, L * L)
  return X

# Ceil top xL predictions to 1, others to zero
def ceil_top_xL_to_one(ref_file_dict, XP, Y, x):
  X_ceiled = np.copy(XP)
  i = -1
  for pdb in sorted(ref_file_dict):
    i = i + 1
    xL = int(x * ref_file_dict[pdb])
    X_ceiled[i, :] = np.zeros(len(XP[i, :]))
    X_ceiled[i, np.argpartition(XP[i, :], -xL)[-xL:]] = 1
  return X_ceiled

def print_detailed_evaluations(dict_l, dict_n, dict_e, PL5, PL2, PL, Y):
  datacount = len(dict_l)
  print("  ID    PDB      L   Nseq   Neff     Nc    L/5  PcL/5  PcL/2   Pc1L    AccL/5    AccL/2      AccL")
  avg_nc  = 0    # average true Nc
  avg_pc_l5  = 0 # average predicted correct L/5
  avg_pc_l2  = 0 # average predicted correct L/2
  avg_pc_1l  = 0 # average predicted correct 1L
  avg_acc_l5 = 0.0
  avg_acc_l2 = 0.0
  avg_acc_1l = 0.0
  list_acc_l5 = []
  list_acc_l2 = []
  list_acc_1l = []
  i = -1
  for pdb in sorted(dict_l):
    i = i + 1
    nc = int(Y[i].sum())
    L = dict_l[pdb]
    L5 = int(L/5)
    L2 = int(L/2)
    pc_l5 = np.logical_and(Y[i], PL5[i, :]).sum()
    pc_l2 = np.logical_and(Y[i], PL2[i, :]).sum()
    pc_1l = np.logical_and(Y[i], PL[i, :]).sum()
    acc_l5 = float(pc_l5) / (float(L5) + epsilon)
    acc_l2 = float(pc_l2) / (float(L2) + epsilon)
    acc_1l = float(pc_1l) / (float(L) + epsilon)
    list_acc_l5.append(acc_l5)
    list_acc_l2.append(acc_l2)
    list_acc_1l.append(acc_1l)
    print(" %3s %6s %6s %6s %6s %6s %6s %6s %6s %6s    %.4f    %.4f    %.4f" % (i, pdb, L, dict_n[pdb], dict_e[pdb], nc, L5, pc_l5, pc_l2, pc_1l, acc_l5, acc_l2, acc_1l))
    avg_nc = avg_nc + nc
    avg_pc_l5 = avg_pc_l5 + pc_l5
    avg_pc_l2 = avg_pc_l2 + pc_l2
    avg_pc_1l = avg_pc_1l + pc_1l
    avg_acc_l5 = avg_acc_l5 + acc_l5
    avg_acc_l2 = avg_acc_l2 + acc_l2
    avg_acc_1l = avg_acc_1l + acc_1l
  avg_nc = int(avg_nc/datacount)
  avg_pc_l5 = int(avg_pc_l5/datacount)
  avg_pc_l2 = int(avg_pc_l2/datacount)
  avg_pc_1l = int(avg_pc_1l/datacount)
  avg_acc_l5 = avg_acc_l5/datacount
  avg_acc_l2 = avg_acc_l2/datacount
  avg_acc_1l = avg_acc_1l/datacount
  print("   Avg                           %6s        %6s %6s %6s    %.4f    %.4f    %.4f" % (avg_nc, avg_pc_l5, avg_pc_l2, avg_pc_1l, avg_acc_l5, avg_acc_l2, avg_acc_1l))
  print ("")
  return (list_acc_l5, list_acc_l2, list_acc_1l)


dist_string = '80'
path_lists = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/lists-test-train/'
pathY         = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/'
pathX         = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/'
l_max=300 # 800 will get memory error
tr_l, tr_n, tr_e, te_l, te_n, te_e = build_dataset_dictionaries(path_lists)


# Make combined dictionaries as well
all_l = te_l.copy()
all_n = te_n.copy()
all_e = te_e.copy()
all_l.update(tr_l)
all_n.update(tr_n)
all_e.update(tr_e)
print('Total Number of Training and Test dataset = ',str(len(all_l)))

sys.stdout.flush()
print('Load all data into memory..',end='')
selected_list = subset_pdb_dict(all_l,   0, l_max, l_max, 'ordered')
print('Loading data sets ..',end='')
(selected_list_1D,selected_list_2D) = get_x_1D_2D_from_this_list(selected_list, pathX, l_max)
print("selected_list_1D.shape: ",selected_list_1D.shape)
print("selected_list_2D.shape: ",selected_list_2D.shape)
print('Loading label sets..')
selected_list_label = get_y_from_this_list(selected_list, pathY, 24, l_max, dist_string)

feature_1D_num = selected_list_1D.shape[2]
feature_2D_num = selected_list_2D.shape[3]
sequence_length = selected_list_1D.shape[1]
DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,1)
DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
#DNCON4_CNN.load_weights(model_weight_out_best) # report error when sequence length changed
DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_1D,selected_list_2D], batch_size= 1)
(list_acc_l5, list_acc_l2, list_acc_1l) = evaluate_prediction(selected_list, all_n, all_e, DNCON4_CNN_prediction, selected_list_label, 24)


