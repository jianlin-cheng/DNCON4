#module load cuda/cuda-9.0.176
#module load cudnn/cudnn-7.1.4-cuda-9.0.176
#export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=""

# srun -p Gpu -N1 -n1 -t 0-02:00 --gres gpu:1  --pty /bin/bash
# cd /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/Test/Distance_Relu2D_sigmoid_MSE/filter64_layers6_inter150_optnadam_ftsize3_batchsize1_he_normal_sigmoid_MSE_1.0/samples

#source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate
#THEANO_FLAGS=floatX=float32,device=gpu  python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/ResNet_arch/scripts/predict_deepResNet_2D.py  eva.list ../model-train-DNCON4_2dRES.json ../model-train-weight-DNCON4_2dRES-best-val.h5  1NEU-A-pred.txt 'nadam' /storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/  'he_normal' 'sigmoid_MSE' 1

import sys
import os
from shutil import copyfile
import platform
from glob import glob

if len(sys.argv) != 10:
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

evalist=sys.argv[1] #
model_file=sys.argv[2] #
model_weight_out_best=sys.argv[3] #
cmap_file=sys.argv[4] #
opt=sys.argv[5] #nadam
feature_dir = sys.argv[6] #/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/
initializer = sys.argv[7]
loss_function = sys.argv[8]
weight_p = float(sys.argv[9])



if sysflag == 'local':
  import tensorflow as tf
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  config = tf.ConfigProto(allow_soft_placement = True)
  tf.GPUOptions(per_process_gpu_memory_fraction = 0.99)
  config.gpu_options.allow_growth = True
  sess= tf.Session(config = config)


dist_string = '80'

path_of_lists='/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/lists-test-train/'
reject_fea_file = '/scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/feature_to_use_lewis.txt'

path_of_Y         =  feature_dir 
path_of_X         = feature_dir
Maximum_length=300 # 800 will get memory error

import time

start_time = time.time()

DNCON4_predict(evalist,model_file,model_weight_out_best,cmap_file,opt,path_of_lists, path_of_Y, path_of_X, Maximum_length,dist_string, reject_fea_file,initializer, loss_function, weight_p)
    
    
DNCON4_predict(evalist,feature_num,model_prefix,win_array,use_bias,hidden_type,nb_filters,nb_layers,model_weight_out_best,cmap_file,opt,path_of_lists, path_of_Y, path_of_X, Maximum_length,dist_string, reject_fea_file    
    
    