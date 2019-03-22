# export  HDF5_USE_FILE_LOCKING=FALSE
# cd /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/Test/cullpdb_Distance_Relu2D_weighted_MSElimited_inloop/filter64_layers6_inter150_optnadam_ftsize3_batchsize1_he_normal_weighted_MSE_limited_1.0/test
# python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/Model_predict_distance.py ../model-train-DNCON4_2dRES.json  ../model-train-weight-DNCON4_2dRES-best-val.h5 ./evadir testlist 
# python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/cmap2rr.py evadir/pred_map/3BFO-B-contact.txt  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/fasta/3BFO-B.fasta  evadir/pred_map/3BFO-B-contact.rr
# python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/cmap2rr_eva.py evadir/pred_map/3BFO-B-contact.txt  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/fasta/3BFO-B.fasta  evadir/pred_map/3BFO-B-contact.rr /storage/htc/bdm/DNCON4/data/cullpdb_dataset/chains/3BFO-B.chn
# python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/dist2image.py evadir/pred_map/3BFO-B-distance.txt  evadir/pred_map/3BFO-B-distance.png
# python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/dist2image.py /storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/3BFO-B.txt evadir/pred_map/3BFO-B-distance_real.png

#python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/distmap2rr.py evadir/pred_map/3BFO-B-distance.txt  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/fasta/3BFO-B.fasta  evadir/pred_map/3BFO-B-distance.rr

# -*- coding: utf-8 -*-
import sys
import os
from shutil import copyfile
import platform
from glob import glob

if len(sys.argv) != 5:
  print('please input the right parameters')
  sys.exit(1)
current_os_name = platform.platform()
print('%s' % current_os_name)


sysflag='lewis'
GLOBAL_PATH='/scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/'
sys.path.insert(0, GLOBAL_PATH+'/lib/')
print (GLOBAL_PATH+'/lib/')
from Model_training import *
from DNCON_lib import *
from Model_construct import *

from Model_construct import _weighted_binary_crossentropy, _weighted_categorical_crossentropy, _weighted_mean_squared_error
import numpy as np
from keras.models import model_from_json,load_model, Sequential, Model
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adagrad, Adamax, Nadam
from keras.utils import CustomObjectScope
from random import randint

model_out=(sys.argv[1]) #15
model_weight_out_best=(sys.argv[2]) #10
CV_dir=(sys.argv[3]) #10
testlist=sys.argv[4] #nadam


#GLOABL_Path = sys.path[0].split('DNCON4')[0]+'DNCON4/'
print("Find gloabl path :", GLOBAL_PATH)
#feature_dir = GLOABL_Path + '/data/badri_training_benchmark/feats/'
###CASP13
#path_of_lists = GLOABL_Path + '/data/CASP13/lists-test-train/'
#path_of_X= GLOABL_Path + '/data/CASP13/feats/'
#path_of_Y= GLOABL_Path + '/data/CASP13/feats/'


### for dncon2 dataset
#feature_dir='/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/'
#path_of_lists='/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/lists-test-train/'
#reject_fea_file = '/scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/feature_to_use_lewis.txt'


#### for cullpdb dataset
feature_dir='/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'
path_of_lists='/storage/htc/bdm/tianqi/DNCON4/data/lists-test-train/'
reject_fea_file = '/scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/feature_to_use_lewis.txt'


path_of_X=feature_dir
path_of_Y=feature_dir

path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/real_dist_map/'
path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'

#path_of_Y_train = '/storage/htc/bdm/DNCON4/feature/map/cullpdb_map/dist_map/'
#path_of_Y_evalu = '/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/'


###DeepCOV
# path_of_lists = GLOABL_Path+'/data/deepcov/lists-test-train/'
# path_of_X = GLOABL_Path + '/data/deepcov/feats/'
# path_of_Y = GLOABL_Path + '/data/deepcov/feats/'
###DNCON2
# path_of_lists = GLOABL_Path + '/data/badri_training_benchmark/lists-test-train/'
# path_of_X = GLOABL_Path + '/data/badri_training_benchmark/feats/'
# path_of_Y = GLOABL_Path + '/data/badri_training_benchmark/feats/'
#reject_fea_path = GLOABL_Path + '/architecture_distance/lib/'
#reject_fea_file = 'feature_to_use_pre.txt' #if feature list set other then will use this reject fea file


model_name = 'DNCON4_2dRES' #'DNCON4_2dCONV' 'DNCON4_2dRES' 'RES'
feature_list = 'other'# ['combine','other', 'ensemble']  # combine will output three map and it combine, other just output one pred
data_list_choose = 'test'# ['train', 'test', 'all']
only_predict_flag = True # if do not have lable set True
Maximum_length = 750
dist_string = "80"
loss_function = _weighted_mean_squared_error(1)

#'binary_crossentropy'
#_weighted_mean_squared_error(1)
#'categorical_crossentropy'


def chkdirs(fn):
    dn = os.path.dirname(fn)
    if not os.path.exists(dn): os.makedirs(dn)



#model_out= "%s/model-train-%s.json" % (CV_dir, model_name)
#model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir, model_name)
#model_weight_epochs = "%s/model_weights/" % (CV_dir)
#model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir, model_name)


# DNCON4 = DeepResnet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,nb_filters,nb_layers,opt,initializer,loss_function,weight_p,weight_n)
with CustomObjectScope({'InstanceNormalization': InstanceNormalization}):
    json_string = open(model_out).read()
    DNCON4 = model_from_json(json_string)

if os.path.exists(model_weight_out_best):
    print("######## Loading existing weights ",model_weight_out_best)
    DNCON4.load_weights(model_weight_out_best)
else:
    print("Please check the best weights\n")
    exit(-1)

model_predict= "%s/pred_map/"%(CV_dir)
chkdirs(model_predict)

#OTHER = reject_fea_path + reject_fea_file
OTHER = reject_fea_file

### loading test list
tr_l = {}
length_dict = {}
with open(path_of_lists + '/L.txt') as f:
  for line in f:
    cols = line.strip().split()
    length_dict[cols[0]] = int(cols[1])
te_l = {}
with open(testlist) as f:
  for line in f:
    te_l[line.strip()] = length_dict[line.strip()]
print ('Data counts:')
print ('Total : ' + str(len(length_dict)))
print ('Test  : ' + str(len(te_l)))
all_l = te_l.copy()        
all_l.update(tr_l)

print('Total Number to predict = ',str(len(all_l)))

##### running validation
pred_history_out = "%s/predict.acc_history" % (CV_dir) 
selected_list = subset_pdb_dict(all_l,   0, Maximum_length, 5000, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset

title = "Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n"
with open(pred_history_out, "w") as myfile:
            myfile.write(title)
            
step_num = 0
####Predict the trainig data set

for key in selected_list:
    value = selected_list[key]
    p1 = {key: value}
    # if if_use_binsize:
    #     Maximum_length = 320
    # else:
    Maximum_length = value
    if len(p1) < 1:
        continue
    print("start predict %s %d" %(key, value))

    other_cmap_distance_file = "%s/%s-distance.txt" % (model_predict, key)
    other_cmap_contact_file = "%s/%s-contact.txt" % (model_predict, key)
    if os.path.isfile(other_cmap_distance_file):
        print("Already exist: ",other_cmap_distance_file)
        continue
    selected_list_2D_other = get_x_2D_from_this_list(p1, path_of_X, Maximum_length,dist_string, OTHER, value)
    if type(selected_list_2D_other) == bool:
            continue
    DNCON4_prediction_other = DNCON4.predict([selected_list_2D_other], batch_size= 1)    
    CMAP = DNCON4_prediction_other.reshape(Maximum_length, Maximum_length)
    CMAP[CMAP>100] = 100 # incase infinity
    CMAP[CMAP<=0] = 0.001 # incase infinity
    Map_UpTrans = np.triu(CMAP, 1).T
    Map_UandL = np.triu(CMAP)
    real_cmap_other = Map_UandL + Map_UpTrans
    DNCON4_CNN_prediction_dist = np.copy(real_cmap_other)
    DNCON4_CNN_prediction_contact = 1/real_cmap_other # convert to confidence
    
    #other_cmap_distance_file = "%s/%s-distance.txt" % (model_predict, key)
    #other_cmap_contact_file = "%s/%s-contact.txt" % (model_predict, key)
    np.savetxt(other_cmap_distance_file, DNCON4_CNN_prediction_dist, fmt='%.4f')
    np.savetxt(other_cmap_contact_file, DNCON4_CNN_prediction_contact, fmt='%.4f')
    selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, Maximum_length, dist_string, lable_type = 'real')# dist_string 80
    DNCON4_CNN_prediction_dist = DNCON4_CNN_prediction_dist.reshape(len(p1), Maximum_length*Maximum_length)
    global_mse, weighted_mse = evaluate_prediction_dist_4(DNCON4_CNN_prediction_dist, selected_list_label_dist)                
    # if just for generate predict map, stop here is fine 
    #if only_predict_flag == True:
    #    continue

    print('Loading label sets..')
    # casp target

    #if list(key)[0] == 'T':
    #    selected_list_label = get_y_from_this_list_casp(p1, path_of_Y_evalu, 0, Maximum_length, dist_string)# dist_string 80
    #else:
    #    selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, Maximum_length, dist_string)# dist_string 80
    DNCON4_CNN_prediction = DNCON4_CNN_prediction_contact.reshape(len(p1), Maximum_length*Maximum_length)
    selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, Maximum_length, dist_string)# dist_string 80
    print(DNCON4_CNN_prediction_contact.shape)
    print(selected_list_label.shape)
    print(p1)
    (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = evaluate_prediction_4(p1, DNCON4_CNN_prediction, selected_list_label, 24)

    val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key,value,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l,global_mse,weighted_mse)
    print("Seq_Name\tSeq_Length\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n")
    print('The best validation accuracy is ',val_acc_history_content)
    with open(pred_history_out, "a") as myfile:
                myfile.write(val_acc_history_content)
    step_num += 1
