##(1) cp -ar /storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/CNN_arch/  /scratch/jh7x3/DNCON4/architecture/
##(2) cd /scratch/jh7x3/DNCON4/architecture/CNN_arch/scripts
## load keras2 
##(3) source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate
## export HDF5_USE_FILE_LOCKING=FALSE
##(4) python


import sys
import os
from shutil import copyfile
import platform

if len(sys.argv) != 11:
  print('please input the right parameters')
  sys.exit(1)
current_os_name = platform.platform()
print('%s' % current_os_name)

#GLOBAL_PATH='/scratch/jh7x3/DNCON4/architecture/CNN_arch/'
if current_os_name == 'Linux-4.15.0-36-generic-x86_64-with-Ubuntu-18.04-bionic': #on local
  GLOBAL_PATH='/mnt/data/zhiye/Python/DNCON4/architecture'
elif current_os_name == 'Linux-3.10.0-862.14.4.el7.x86_64-x86_64-with-centos-7.5.1804-Core': #on lewis
  GLOBAL_PATH=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
else:
  print('Please check current operate system!')
  sys.exit(1)

print (os.path.dirname(GLOBAL_PATH))
sys.path.insert(0, GLOBAL_PATH+'/lib/')
from Data_loading import load_train_test_data_padding_with_interval_2D
from Model_training import DNCON4_1d2dconv_train_win_filter_layer_opt_fast_2Donly


inter=int(sys.argv[1]) #15
nb_filters=int(sys.argv[2]) #10
nb_layers=int(sys.argv[3]) #10
opt=sys.argv[4] #nadam
filtsize=sys.argv[5] #6_10
out_epoch=int(sys.argv[6]) #100
in_epoch=int(sys.argv[7]) #3
feature_dir = sys.argv[8] #/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/
outputdir = sys.argv[9] 
batchsize = int(sys.argv[10])


#inter=15
#nb_filters=10
#nb_layers=10
#opt='nadam'
#filtsize='6'
#out_epoch=1
#in_epoch=1
#feature_dir = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/'
#outputdir = '/scratch/jh7x3/DNCON4/architecture/CNN_arch/test_out'
#batchsize =5

CV_dir=outputdir+'/filter'+str(nb_filters)+'_layers'+str(nb_layers)+'_inter'+str(inter)+'_opt'+str(opt)+'_ftsize'+str(filtsize)+'_batchsize'+str(batchsize)

lib_dir=GLOBAL_PATH+'/lib/'

import tensorflow as tf
config = tf.ConfigProto(allow_soft_placement = True)
tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
config.gpu_options.allow_growth = True
sess= tf.Session(config = config)

filetsize_array = list(map(int,filtsize.split("_")))

if not os.path.exists(CV_dir):
    os.makedirs(CV_dir)



dist_string = '80'
path_of_lists = os.path.dirname(GLOBAL_PATH)+'/data/badri_training_benchmark/lists-test-train_sample20/'
#path_of_lists = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/lists-test-train_example/'
path_of_Y         =  feature_dir
path_of_X         = feature_dir
Maximum_length=300 # 800 will get memory error


train_datafile=path_of_lists + '/train.lst';
val_datafile=path_of_lists + '/test.lst';

import time
data_all_dict_padding = load_train_test_data_padding_with_interval_2D(train_datafile, feature_dir,inter,5000,0,dist_string)
testdata_all_dict_padding = load_train_test_data_padding_with_interval_2D(val_datafile, feature_dir, inter,5000,0,dist_string)  


start_time = time.time()
DNCON4_1d2dconv_train_win_filter_layer_opt_fast_2Donly(data_all_dict_padding,testdata_all_dict_padding,CV_dir, feature_dir,"DNCON4_2dCONV",out_epoch,in_epoch,inter,5000,filetsize_array,True,'sigmoid',nb_filters,nb_layers,opt,lib_dir, batchsize,path_of_lists,path_of_Y, path_of_X,Maximum_length,dist_string)

print("--- %s seconds ---" % (time.time() - start_time))
