import sys
import os
from shutil import copyfile
import platform

if len(sys.argv) != 12:
  print('please input the right parameters')
  sys.exit(1)
current_os_name = platform.platform()
print('%s' % current_os_name)


#GLOBAL_PATH='/scratch/jh7x3/DNCON4/architecture/CNN_arch/'
if current_os_name == 'Linux-4.15.0-42-generic-x86_64-with-Ubuntu-18.04-bionic': #on local
  GLOBAL_PATH='/mnt/data/zhiye/Python/DNCON4/architecture'
  sysflag='local'
elif current_os_name == 'Linux-3.10.0-862.14.4.el7.x86_64-x86_64-with-centos-7.5.1804-Core': #on lewis
  GLOBAL_PATH=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
  sysflag='lewis'
else:
  print('Please check current operate system!')
  sys.exit(1)

print (os.path.dirname(GLOBAL_PATH))
sys.path.insert(0, GLOBAL_PATH+'/lib/')
from Data_loading import load_train_test_data_padding_with_interval, load_train_test_data_padding_with_interval_2D
from Model_training import *


inter=int(sys.argv[1]) #15
nb_filters=int(sys.argv[2]) #10
nb_layers=int(sys.argv[3]) #10
opt=sys.argv[4] #nadam
filtsize=sys.argv[5] #6_10
out_epoch=int(sys.argv[6]) #100
in_epoch=int(sys.argv[7]) #3
feature_dir = sys.argv[8] #/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/
outputdir = sys.argv[9] 
acclog_dir = sys.argv[10]
batchsize = int(sys.argv[11])


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

if sysflag == 'local':
  import tensorflow as tf
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  config = tf.ConfigProto(allow_soft_placement = True)
  tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)
  config.gpu_options.allow_growth = True
  sess= tf.Session(config = config)

filetsize_array = list(map(int,filtsize.split("_")))

if not os.path.exists(CV_dir):
    os.makedirs(CV_dir)
# else:
    # print ('File exit, quit!')
    # sys.exit(1)

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

def chkfiles(fn):
  if os.path.exists(fn):
    return True 
  else:
    return False

dist_string = '80'
if sysflag == 'local':
  path_of_lists = os.path.dirname(GLOBAL_PATH)+'/data/badri_training_benchmark/lists-test-train_sample120/'
  reject_fea_file =  GLOBAL_PATH+'/lib/feature_to_use.txt'
elif sysflag == 'lewis':
  path_of_lists = os.path.dirname(GLOBAL_PATH)+'/data/badri_training_benchmark/lists-test-train_sample120/'
  reject_fea_file =  GLOBAL_PATH+'/lib/feature_to_use_lewis.txt'
path_of_Y         =  feature_dir
path_of_X         = feature_dir
Maximum_length=300 # 800 will get memory error


train_datafile=path_of_lists + '/train.lst'
val_datafile=path_of_lists + '/test.lst'

import time

data_all_dict_padding = load_train_test_data_padding_with_interval_2D(train_datafile, feature_dir,inter,5000,0,dist_string, reject_fea_file)
testdata_all_dict_padding = load_train_test_data_padding_with_interval_2D(val_datafile, feature_dir, inter,5000,0,dist_string, reject_fea_file)  

start_time = time.time()
best_acc=DNCON4_1d2dconv_train_win_filter_layer_opt_fast_2Donly(data_all_dict_padding,testdata_all_dict_padding,CV_dir, feature_dir,"DNCON4_2dINCEP",out_epoch,in_epoch,inter,5000,filetsize_array,True,'sigmoid',nb_filters,nb_layers,opt,lib_dir, batchsize,path_of_lists,path_of_Y, path_of_X,Maximum_length,dist_string, reject_fea_file)

model_prefix = "INCEP"
acc_history_out = "%s/%s.acc_history" % (acclog_dir, model_prefix)
chkdirs(acc_history_out)
if chkfiles(acc_history_out):
    print ('acc_file_exist,pass!')
    pass
else:
    print ('create_acc_file!')
    with open(acc_history_out, "w") as myfile:
        myfile.write("time\t netname\t filternum\t layernum\t kernelsize\t batchsize\t accuracy\n")

time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
acc_history_content = "%s\t %s\t %s\t %s\t %s\t %s\t %.4f\n" % (time_str, model_prefix, str(nb_filters),str(nb_layers),str(filtsize),str(batchsize),best_acc)
with open(acc_history_out, "a") as myfile: myfile.write(acc_history_content) 
print("--- %s seconds ---" % (time.time() - start_time))

