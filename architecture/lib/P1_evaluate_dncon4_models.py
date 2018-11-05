#!/usr/bin/python
# Badri Adhikari, 6-15-2017
# Main training script

# srun -p Interactive --mem 30G -t 0-02:00 --pty --licenses=matlab:1 /bin/bash
# source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate

# run dncon4 weight: python /storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/lib/P1_evaluate_dncon4_models.py 80 /storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/outputs/Incep_arch/model-train-DNCON4_1d2dINCEP.json /storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/outputs/Incep_arch/model-train-weight-DNCON4_1d2dINCEP-epoch8.h5


from keras.models import model_from_json
import keras.backend as K
from keras.datasets import mnist
from keras.engine.topology import Layer
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, Lambda, add, concatenate,ConvLSTM2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.constraints import maxnorm

from keras import utils
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, MaxPooling1D, AveragePooling1D,UpSampling1D, LSTM,Average
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.activations import tanh, softmax

import tensorflow as tf

import shutil
import sys
project_root = '/mnt/data/zhiye/Python/DNCON4/data/badri_training_benchmark/'
sys.path.insert(0, '/mnt/data/zhiye/Python/DNCON4/architecture/lib')
from DNCON_lib import *
from Model_construct import DeepInception_with_paras

"""
win_array=[5]
feature_1D_num = 22
feature_2D_num = 18 
sequence_length = 300
use_bias= True
hidden_type= 'sigmoid'
nb_filters=33
nb_layers = 4
opt='nadam'
batch_size_train_new=1
DNCON4_CNN = DeepInception_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt,batch_size_train_new)
model_json = DNCON4_CNN.to_json()
with open('test.json', "w") as json_file:
    json_file.write(model_json)
"""

def print_feature_summary(X):
	print('FeatID         Avg        Med        Max        Sum        Avg[30]    Med[30]    Max[30]    Sum[30]')
	for ii in range(0, len(X[0, 0, 0, :])):
		(m,s,a,d) = (X[0, :, :, ii].flatten().max(), X[0, :, :, ii].flatten().sum(), X[0, :, :, ii].flatten().mean(), np.median(X[0, :, :, ii].flatten()))
		(m30,s30,a30, d30) = (X[0, 30, :, ii].flatten().max(), X[0, 30, :, ii].flatten().sum(), X[0, 30, :, ii].flatten().mean(), np.median(X[0, 30, :, ii].flatten()))
		print(' Feat%2s %10.4f %10.4f %10.4f %10.1f     %10.4f %10.4f %10.4f %10.4f' %(ii, a, d, m, s, a30, d30, m30, s30))

class generatePairwiseF(Layer):
    '''
        (l,n) -> (l*l,3n)
    '''
    def __init__(self, output_shape, batch_size, **kwargs):
        self._output_shape = tuple(output_shape)
        self._batch_size = batch_size
        #super(generatePairwiseF, self).__init__()
        super(generatePairwiseF, self).__init__(**kwargs)
    
    def compute_output_shape(self,input_shape):
        shape = list(input_shape)
        assert len(shape) == 3  # only valid for 3D tensors
        return (input_shape[0],self._output_shape[0], self._output_shape[1], self._output_shape[2])
    
    def call(self, x, mask=None):
        dim = x.shape.as_list()
        if dim[0] is None:
            print(self._batch_size)
        else:
            self._batch_size = dim[0]
            print(self._batch_size)
        for i in range(0, self._batch_size):
            orign = x[i]
            temp = tf.fill([orign.shape[0],orign.shape[0],orign.shape[1]], 1.0)
            first = tf.multiply(temp, orign)
            second = tf.transpose(first, [1,0,2])
            avg = tf.div(tf.add(first, second), 2)
            combine = tf.concat([first, second, avg], axis=-1)
            expand = tf.reshape(combine, [1, combine.shape[0],combine.shape[1],combine.shape[2]])
            if (i==0):
                output = expand
            else:
                output = tf.concat([output, expand], axis=0)
        outputnew =  K.reshape(output,(-1,self._output_shape[0],self._output_shape[1],self._output_shape[2]))
        return outputnew
    
    def get_config(self):
        config = {'batch_size': self._batch_size,'output_shape': self._output_shape}
        base_config = super(generatePairwiseF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

dist_string   = sys.argv[1]
file_model   = sys.argv[2]
file_weights   = sys.argv[3]
path_of_X         = project_root+'/feats/'
path_of_Y         = project_root+'/feats/'
path_lists    = project_root+'/lists-test-train/'

print('')
print('Parameters:')
print('dist_string   ' + dist_string)
print('pathX         ' + path_of_X)
print('pathY         ' + path_of_Y)
print('')


#Loading Validation data into Memory
#Maximum_length=300 # 800 will get memory error
tr_l, tr_n, tr_e, te_l, te_n, te_e = build_dataset_dictionaries(path_lists)
# Make combined dictionaries as well
all_l = te_l.copy()
all_n = te_n.copy()
all_e = te_e.copy()
all_l.update(tr_l)
all_n.update(tr_n)
all_e.update(tr_e)
print('Total Number of Training and Test dataset = ',str(len(all_l)))

Maximum_length = 300

sys.stdout.flush()
print('Load all test data into memory..',end='')
selected_list = subset_pdb_dict(te_l,   0, Maximum_length, Maximum_length, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
print('Loading data sets ..',end='')
(selected_list_1D,selected_list_2D) = get_x_1D_2D_from_this_list(selected_list, path_of_X, Maximum_length,dist_string)
print("selected_list_1D.sum: ",np.sum(selected_list_1D))
print("selected_list_2D.sum: ",np.sum(selected_list_2D))
print("selected_list_1D.shape: ",selected_list_1D.shape)
print("selected_list_2D.shape: ",selected_list_2D.shape)
print('Loading label sets..')
selected_list_label = get_y_from_this_list(selected_list, path_of_Y, 24, Maximum_length, dist_string)
feature_1D_num_vali = selected_list_1D.shape[2]
feature_2D_num_vali = selected_list_2D.shape[3]
sequence_length = selected_list_1D.shape[1]

json_file_model = open(file_model, 'r')
loaded_model_json = json_file_model.read()
json_file_model.close()
DNCON4_CNN = model_from_json(loaded_model_json, custom_objects={'generatePairwiseF': generatePairwiseF})
DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer='nadam')
DNCON4_CNN.load_weights(file_weights)
DNCON4_CNN_prediction = DNCON4_CNN.predict([selected_list_1D,selected_list_2D], batch_size= 1)
(list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = evaluate_prediction(selected_list, all_n, all_e, DNCON4_CNN_prediction, selected_list_label, 24)


