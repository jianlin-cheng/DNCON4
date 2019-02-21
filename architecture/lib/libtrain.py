#!/usr/bin/python
# Badri Adhikari, 6-15-2017
# Subroutines used in training and testing

import shutil
from libcommon import * 

# Training hyperparameters
def read_train_param(file_config):
	if not os.path.isfile(file_config):
		print ('Error! Could not find config file ' + file_config)
		sys.exit(1)
	train_param = {}
	with open(file_config) as f:
		for line in f:
			if line.startswith('#'):
				continue
			if len(line) < 2:
				continue
			cols = line.strip().split()
			if len(cols) < 2:
				print ('Error! Config file ' + file_config + ' line ' + line + '??')
				sys.exit(1)
			if cols[0] == 'optimizer':
				train_param[cols[0]] = cols[1]
			else:
				train_param[cols[0]] = int(cols[1])
	print ('')
	print ('Read training parameters:')
	for k, v in sorted(train_param.items()):
		print ("%-15s : %-3s" %(k, v))
	print ('')
	return train_param

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

def print_detailed_evaluations(dict_l, dict_n, dict_e, PL5, PL2, PL, Y):
	datacount = len(dict_l)
	print "  ID    PDB      L   Nseq   Neff     Nc    L/5  PcL/5  PcL/2   Pc1L    AccL/5    AccL/2      AccL"
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
		print " %3s %6s %6s %6s %6s %6s %6s %6s %6s %6s    %.4f    %.4f    %.4f" % (i, pdb, L, dict_n[pdb], dict_e[pdb], nc, L5, pc_l5, pc_l2, pc_1l, acc_l5, acc_l2, acc_1l)
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
	print "   Avg                           %6s        %6s %6s %6s    %.4f    %.4f    %.4f" % (avg_nc, avg_pc_l5, avg_pc_l2, avg_pc_1l, avg_acc_l5, avg_acc_l2, avg_acc_1l)
	print ("")
	return (list_acc_l5, list_acc_l2, list_acc_1l)

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

def get_x_from_this_list(selected_ids, path, l_max):
	xcount = len(selected_ids)
	sample_pdb = ''
	for pdb in selected_ids:
		sample_pdb = pdb
		break
	print path + 'X-'  + sample_pdb + '.txt'
	x = getX(path + 'X-'  + sample_pdb + '.txt', l_max)
	F = len(x[0, 0, :])
	X = np.zeros((xcount, l_max, l_max, F))
	i = 0
	for pdb in sorted(selected_ids):
		T = getX(path + 'X-'  + pdb + '.txt', l_max)
		if len(T[0, 0, :]) != F:
			print 'ERROR! Feature length of ' + sample_pdb + ' not equal to ' + pdb
		X[i, :, :, :] = T
		i = i + 1
	return X

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

def train_on_this_X_Y (model_arch, train_param, X, Y, out_file_weights):
	print ''
	print 'X Train shape : ' + str(X.shape)
	print 'Y Train shape : ' + str(Y.shape)
	print ''
	model = build_model_for_this_input_shape(model_arch, X)
	if os.path.isfile(out_file_weights):
		print 'Loading previously saved weights..'
		print ''
		model.load_weights(out_file_weights)
	else:
		print model.summary()
	print ''
	print 'Compiling model..'
	model.compile(loss = 'binary_crossentropy', optimizer = train_param['optimizer'], metrics = ['accuracy'])
	print ''
	print 'Fitting model..'
	model.fit(X, Y, verbose = 1, batch_size = train_param['batch_size'], nb_epoch = train_param['inner_epochs'])
	model.save_weights(out_file_weights)

def evaluate_on_this_X_Y (model_arch, file_weights, file_dict, X, Y, contact_selection, eval_type):
	model = build_model_for_this_input_shape(model_arch, X)
	model.load_weights(file_weights)
	P1 = model.predict(X)
	P2 = floor_lower_left_to_zero(P1, contact_selection)
	datacount = len(Y[:, 0])
	L = int(math.sqrt(len(Y[0, :])))
	Y1 = floor_lower_left_to_zero(Y, contact_selection)
	list_acc_l5 = []
	list_acc_l2 = []
	list_acc_1l = []
	if eval_type == 'top-nc':
		P3 = ceil_top_Nc_to_one(P2, Y1)
		avg_acc = print_detailed_evaluations_top_Nc(file_dict, P3, Y1)
	elif eval_type == 'top-l-all':
		P3L5 = ceil_top_xL_to_one(file_dict, P2, Y, 0.2)
		P3L2 = ceil_top_xL_to_one(file_dict, P2, Y, 0.5)
		P31L = ceil_top_xL_to_one(file_dict, P2, Y, 1)
		(list_acc_l5, list_acc_l2, list_acc_1l) = print_detailed_evaluations(file_dict, P3L5, P3L2, P31L, Y)
	elif eval_type == 'top-l5':
		P3 = ceil_top_L5_to_one(file_dict, P2, Y)
		avg_acc = print_detailed_evaluations_top_L5(file_dict, P3, Y)
	return (list_acc_l5, list_acc_l2, list_acc_1l)
