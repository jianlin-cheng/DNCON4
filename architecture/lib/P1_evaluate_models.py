#!/usr/bin/python
# Badri Adhikari, 6-15-2017
# Main training script

# srun -p Interactive --mem 30G -t 0-02:00 --pty --licenses=matlab:1 /bin/bash
# source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate

# run dncon2 weight: python /storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/lib/P1_evaluate_models.py 80 /storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/best-weights/stage1-80A.hdf5


import shutil
import sys
project_root = '/scratch/jh7x3/DNCON4/data/badri_training_benchmark/'
sys.path.insert(0, project_root + 'lib_python3')
from libtrain import *

dist_string   = sys.argv[1]
weightfile   = sys.argv[2]
pathX         = project_root+'feats/'
pathY         = project_root+'feats/'
path_lists    = project_root+'lists-test-train/'

print('')
print('Parameters:')
print('dist_string   ' + dist_string)
print('pathX         ' + pathX)
print('pathY         ' + pathY)
print('')

model_arch = read_model_arch(project_root + '/scripts/model-arch.config')
train_param = read_train_param(project_root + '/scripts/train-param.config')
tr_l, tr_n, tr_e, te_l, te_n, te_e = build_dataset_dictionaries(path_lists)

# Make combined dictionaries as well
all_l = te_l.copy()
all_n = te_n.copy()
all_e = te_e.copy()
all_l.update(tr_l)
all_n.update(tr_n)
all_e.update(tr_e)
print('Total Number of Training and Test dataset = ' + str(len(all_l)))

def print_detailed_accuracy_on_this_data(id_string, file_weights):
	print ('')
	all_list_acc_l5 = []
	all_list_acc_l2 = []
	all_list_acc_1l = []
	for group in range(0, 3):
		L = LRT1
		X = XRT1
		Y = YRT1
		if 'TRAIN' in id_string:
			print('Printing detailed results for TRAIN group ' + str(group))
			if group == 0:
				P = make_prediction(model_arch, file_weights, XRT1)
				(list_acc_l5, list_acc_l2, list_acc_1l) = evaluate_prediction(LRT1, all_n, all_e, P, YRT1, 24)
				all_list_acc_l5.extend(list_acc_l5)
				all_list_acc_l2.extend(list_acc_l2)
				all_list_acc_1l.extend(list_acc_1l)
			if group == 1:
				P = make_prediction(model_arch, file_weights, XRT2)
				(list_acc_l5, list_acc_l2, list_acc_1l) = evaluate_prediction(LRT2, all_n, all_e, P, YRT2, 24)
				all_list_acc_l5.extend(list_acc_l5)
				all_list_acc_l2.extend(list_acc_l2)
				all_list_acc_1l.extend(list_acc_1l)
			if group == 2:
				P = make_prediction(model_arch, file_weights, XRT3)
				(list_acc_l5, list_acc_l2, list_acc_1l) = evaluate_prediction(LRT3, all_n, all_e, P, YRT3, 24)
				all_list_acc_l5.extend(list_acc_l5)
				all_list_acc_l2.extend(list_acc_l2)
				all_list_acc_1l.extend(list_acc_1l)
		if 'TEST' in id_string:
			print('Printing detailed results for TEST group ' + str(group))
			if group == 0:
				P = make_prediction(model_arch, file_weights, XTE1)
				print("prediction shape",P.shape)
				print("prediction: ",P)
				for ix in range(0,len(YTE1)):
					print ("test label :",ix," ",np.sum(YTE1[ix]))
				(list_acc_l5, list_acc_l2, list_acc_1l) = evaluate_prediction(LTE1, all_n, all_e, P, YTE1, 24)
				all_list_acc_l5.extend(list_acc_l5)
				all_list_acc_l2.extend(list_acc_l2)
				all_list_acc_1l.extend(list_acc_1l)
			if group == 1:
				P = make_prediction(model_arch, file_weights, XTE2)
				print("prediction shape",P.shape)
				print("prediction: ",P)
				for ix in range(0,len(YTE2)):
						print ("test label :",ix," ",np.sum(YTE2[ix]))
				
				(list_acc_l5, list_acc_l2, list_acc_1l) = evaluate_prediction(LTE2, all_n, all_e, P, YTE2, 24)
				all_list_acc_l5.extend(list_acc_l5)
				all_list_acc_l2.extend(list_acc_l2)
				all_list_acc_1l.extend(list_acc_1l)
			if group == 2:
				P = make_prediction(model_arch, file_weights, XTE3)
				print("prediction shape",P.shape)
				print("prediction: ",P)
				for ix in range(0,len(YTE3)):
					print ("test label :",ix," ",np.sum(YTE3[ix]))
				(list_acc_l5, list_acc_l2, list_acc_1l) = evaluate_prediction(LTE3, all_n, all_e, P, YTE3, 24)
				all_list_acc_l5.extend(list_acc_l5)
				all_list_acc_l2.extend(list_acc_l2)
				all_list_acc_1l.extend(list_acc_1l)
	acc_l5 = sum(all_list_acc_l5) / len(all_list_acc_l5)
	acc_l2 = sum(all_list_acc_l2) / len(all_list_acc_l2)
	acc_1l = sum(all_list_acc_1l) / len(all_list_acc_1l)
	print('----------------------------------------------------------------------------------------------')
	print('Cycle DataSet      Acc-L/5  Acc-L/2  Acc-L')
	print('' + id_string + ' %.3f    %.3f    %.3f' %(acc_l5, acc_l2, acc_1l))
	print('----------------------------------------------------------------------------------------------')

def print_feature_summary(X):
	print('FeatID         Avg        Med        Max        Sum        Avg[30]    Med[30]    Max[30]    Sum[30]')
	for ii in range(0, len(X[0, 0, 0, :])):
		(m,s,a,d) = (X[0, :, :, ii].flatten().max(), X[0, :, :, ii].flatten().sum(), X[0, :, :, ii].flatten().mean(), np.median(X[0, :, :, ii].flatten()))
		(m30,s30,a30, d30) = (X[0, 30, :, ii].flatten().max(), X[0, 30, :, ii].flatten().sum(), X[0, 30, :, ii].flatten().mean(), np.median(X[0, 30, :, ii].flatten()))
		print(' Feat%2s %10.4f %10.4f %10.4f %10.1f     %10.4f %10.4f %10.4f %10.4f' %(ii, a, d, m, s, a30, d30, m30, s30))

"""
print('Load all Training data into memory..')
LTR1 = subset_pdb_dict(tr_l,   0, 100, 4000, 'random')
LTR2 = subset_pdb_dict(tr_l, 100, 200, 4000, 'random')
LTR3 = subset_pdb_dict(tr_l, 200, 300, 4000, 'random')
print('Loading sets X1, X2, and X3..')
XTR1 = get_x_from_this_list(LTR1, pathX, 100)
XTR2 = get_x_from_this_list(LTR2, pathX, 200)
XTR3 = get_x_from_this_list(LTR3, pathX, 300)
print("XTR1.shape: ",XTR1.shape)
print("XTR2.shape: ",XTR2.shape)
print("XTR3.shape: ",XTR3.shape)
print ('Loading Y1, Y2, and Y3 ..')
YTR1 = get_y_from_this_list(LTR1, pathY, 0, 100, dist_string)
YTR2 = get_y_from_this_list(LTR2, pathY, 0, 200, dist_string)
YTR3 = get_y_from_this_list(LTR3, pathY, 0, 300, dist_string)

sys.stdout.flush()
print('Load representative Training data into memory..')
LRT1 = subset_pdb_dict(tr_l,   0, 100, 100, 'ordered')
LRT2 = subset_pdb_dict(tr_l, 100, 200, 100, 'ordered')
LRT3 = subset_pdb_dict(tr_l, 200, 300, 100, 'ordered')
print('Loading sets X1, X2, and X3..')
XRT1 = get_x_from_this_list(LRT1, pathX, 100)
XRT2 = get_x_from_this_list(LRT2, pathX, 200)
XRT3 = get_x_from_this_list(LRT3, pathX, 300)
print("XRT1.shape: ",XRT1.shape)
print("XRT2.shape: ",XRT2.shape)
print("XRT3.shape: ",XRT3.shape)
print('Loading Y1, Y2, and Y3 ..')
YRT1 = get_y_from_this_list(LRT1, pathY, 24, 100, dist_string)
YRT2 = get_y_from_this_list(LRT2, pathY, 24, 200, dist_string)
YRT3 = get_y_from_this_list(LRT3, pathY, 24, 300, dist_string)
"""

sys.stdout.flush()
print ('Load Test data into memory..')
LTE1 = subset_pdb_dict(te_l,   0, 300, 4000, 'ordered')
#LTE2 = subset_pdb_dict(te_l, 100, 200, 4000, 'ordered')
#LTE3 = subset_pdb_dict(te_l, 200, 300, 4000, 'ordered')
print ('Loading sets X1, X2, and X3..')
XTE1 = get_x_from_this_list(LTE1, pathX, 300)
#XTE2 = get_x_from_this_list(LTE2, pathX, 200)
#XTE3 = get_x_from_this_list(LTE3, pathX, 300)
print ("XTE1.shape: ",XTE1.shape)
#print ("XTE2.shape: ",XTE2.shape)
#print ("XTE3.shape: ",XTE3.shape)
#print ('Loading Y1, Y2, and Y3 ..')
YTE1 = get_y_from_this_list(LTE1, pathY, 24, 300, dist_string)
#YTE2 = get_y_from_this_list(LTE2, pathY, 24, 200, dist_string)
#YTE3 = get_y_from_this_list(LTE3, pathY, 24, 300, dist_string)


# cycle the training groups during training
def next_group(current_group):
	if current_group == 0:
		return 1
	if current_group == 1:
		return 2
	if current_group == 2:
		return 0

#os.system('rm -f *.hdf5')  # changed by jie
group = 0
#for cyc in range (0, train_param['outer_epochs']):
#for cyc in range (0, 100):
sys.stdout.flush()
if not os.path.exists(weightfile):
	print('Skipping ' + weightfile)

print ('Evaluate on full Test data..')
#print_detailed_accuracy_on_this_data(' ALL-TEST    ', weightfile)
P = make_prediction(model_arch, weightfile, XTE1)
(list_acc_l5, list_acc_l2, list_acc_1l) = evaluate_prediction(LTE1, all_n, all_e, P, YTE1, 24)