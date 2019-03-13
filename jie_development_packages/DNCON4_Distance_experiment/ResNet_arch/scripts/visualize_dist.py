import sys
# cd /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/Test/Distance_Relu2D_sigmoid_MSE/filter64_layers6_inter150_optnadam_ftsize3_batchsize1_he_normal_sigmoid_MSE_1.0/samples

#source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate
#THEANO_FLAGS=floatX=float32,device=cpu  python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/ResNet_arch/scripts/visualize_dist.py  eva.list /storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/real_dist/1NEU-A.txt 1NEU-A-true.png
#THEANO_FLAGS=floatX=float32,device=cpu  python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/ResNet_arch/scripts/visualize_dist.py  eva.list ./1NEU-A-pred.txt 1NEU-A-sMSE-predict.png

#THEANO_FLAGS=floatX=float32,device=cpu  python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/ResNet_arch/scripts/visualize_dist.py  eva.list /storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/real_dist/1D7P-M.txt 1D7P-M-true.png
#THEANO_FLAGS=floatX=float32,device=cpu  python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/ResNet_arch/scripts/visualize_dist.py  eva.list ./1D7P-M-pred.txt 1D7P-M-sMSE-predict.png

#THEANO_FLAGS=floatX=float32,device=cpu  python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/ResNet_arch/scripts/visualize_dist.py  eva.list /storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/real_dist/1UXZ-A.txt 1UXZ-A-true.png
#THEANO_FLAGS=floatX=float32,device=cpu  python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/ResNet_arch/scripts/visualize_dist.py  eva.list ./1UXZ-A-pred.txt 1UXZ-A-sMSE-predict.png


import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if len(sys.argv) != 4:
  print('please input the right parameters')
  sys.exit(1)


evalist=sys.argv[1] #
dist_file=sys.argv[2] #
outimage=sys.argv[3] #

GLOBAL_PATH='/scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/'
sys.path.insert(0, GLOBAL_PATH+'/lib/')
print (GLOBAL_PATH+'/lib/')
#from DNCON_lib import *

path_of_lists = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/lists-test-train/'
dist_lable_path = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/real_dist/'
# dist_lable_path = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/bin_class/'
#dist_chart_path = '/mnt/data/zhiye/Python/DNCON4/architecture_distance/Test/'
# tr_l = build_dataset_dictionaries_train(path_of_lists)
length_dict = {}
with open(path_of_lists + 'L.txt') as f:
  for line in f:
    cols = line.strip().split()
    length_dict[cols[0]] = int(cols[1])
te_l = {}
with open(evalist) as f:
  for line in f:
    te_l[line.strip()] = length_dict[line.strip()]
# tr_l = {'1A34-A':147, '1A6M-A': 151}
all_Y =[]
#dist_pic_path = dist_chart_path + '/dist_pic/'
for key in te_l:
    value = te_l[key]
    # dist_file = dist_lable_path + 'Y80-' + key + '.txt'
    #dist_file = dist_lable_path + key + '.txt'
    dist_pic_file = outimage
    Y = np.zeros((value, value))
    L = value
    i = 0
    with open(dist_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            this_line = line.strip().split()
            if len(this_line) != value:
                print("\nThis_line = %i, L = %i, Lable file %s error!\n" % (len(this_line), value, dist_file))
                Y = [0]
            Y[i, 0:L] = np.asarray(this_line)
            i = i + 1
    print("process %s\n" % key)
    # all_Y.append(Y)
    plt.imshow(Y)
    plt.savefig(dist_pic_file)
    plt.close()