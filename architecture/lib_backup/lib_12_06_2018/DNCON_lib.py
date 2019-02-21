# -*- coding: utf-8 -*-


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
from Data_loading import getX_1D_2D,getX_2D_format


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
      Y[i, 0:L] = np.asarray(this_line)
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




def get_x_1D_2D_from_this_list(selected_ids, feature_dir, l_max,dist_string, reject_fea_file='None'):
  xcount = len(selected_ids)
  sample_pdb = ''
  for pdb in selected_ids:
    sample_pdb = pdb
    break
  featurefile =feature_dir + 'X-'  + sample_pdb + '.txt'
  #print(featurefile)
  ### load the data
  (featuredata,feature_index_all_dict) = getX_1D_2D(featurefile, reject_fea_file=reject_fea_file)     
  ### merge 1D data to L*m
  ### merge 2D data to  L*L*n
  feature_1D_all=[]
  feature_2D_all=[]
  for key in sorted(featuredata.keys()):
      featurename = feature_index_all_dict[key]
      feature = featuredata[key]
      feature = np.asarray(feature)
      #print("keys: ", key, " featurename: ",featurename, " feature_shape:", feature.shape)
      #print "keys: ", key, ": ", featuredata[key].shape
      
      if feature.shape[0] == feature.shape[1]:
        feature_2D_all.append(feature)
      else:
        feature_1D_all.append(feature)
  
  fea_len = feature_2D_all[0].shape[0]
  F_2D = len(feature_2D_all)
  
  X_2D_tmp = np.zeros((fea_len, fea_len, F_2D))
  for m in range (0, F_2D):
    X_2D_tmp[0:fea_len, 0:fea_len, m] = feature_2D_all[m]
    
  
  F_1D = len(feature_1D_all)
  
  X_1D_tmp = np.zeros((fea_len, F_1D))
  for m in range (0, F_1D):
    X_1D_tmp[0:fea_len, m] = feature_1D_all[m]
  
  
  feature_1D_all_complete =  X_1D_tmp
  #feature_1D_all_complete.shape #(123, 22)
  feature_2D_all_complete =  X_2D_tmp
  #feature_2D_all_complete.shape #(123, 123, 18)
  fea_len = feature_2D_all_complete.shape[0]
  F_1D = len(feature_1D_all_complete[0, :])
  F_2D = len(feature_2D_all_complete[0, 0, :])
  X_1D = np.zeros((xcount, l_max, F_1D))
  X_2D = np.zeros((xcount, l_max, l_max, F_2D))
  pdb_indx = 0
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
      (featuredata,feature_index_all_dict) = getX_1D_2D(featurefile, reject_fea_file=reject_fea_file)     
      ### merge 1D data to L*m
      ### merge 2D data to  L*L*n
      feature_1D_all=[]
      feature_2D_all=[]
      for key in sorted(featuredata.keys()):
          featurename = feature_index_all_dict[key]
          feature = featuredata[key]
          feature = np.asarray(feature)
          #print("keys: ", key, " featurename: ",featurename, " feature_shape:", feature.shape)
          #print "keys: ", key, ": ", featuredata[key].shape
          
          if feature.shape[0] == feature.shape[1]:
            feature_2D_all.append(feature)
          else:
            feature_1D_all.append(feature)
      
      fea_len = feature_2D_all[0].shape[0]
      F_2D = len(feature_2D_all)
      
      X_2D_tmp = np.zeros((fea_len, fea_len, F_2D))
      for m in range (0, F_2D):
        X_2D_tmp[0:fea_len, 0:fea_len, m] = feature_2D_all[m]
        
      
      F_1D = len(feature_1D_all)
      
      X_1D_tmp = np.zeros((fea_len, F_1D))
      for m in range (0, F_1D):
        X_1D_tmp[0:fea_len, m] = feature_1D_all[m]
      
      
      feature_1D_all_complete =  X_1D_tmp
      #feature_1D_all_complete.shape #(123, 22)
      feature_2D_all_complete =  X_2D_tmp
      #feature_2D_all_complete.shape #(123, 123, 18)
      if len(feature_1D_all_complete[0, :]) != F_1D:
        print('ERROR! 1D Feature length of ',sample_pdb,' not equal to ',pdb_name)
        exit;

      ### expand to lmax
      if feature_1D_all_complete.shape[0] < l_max:
        L = feature_1D_all_complete.shape[0]
        F = feature_1D_all_complete.shape[1]
        X_tmp = np.zeros((l_max, F))
        for i in range (0, F):
          X_tmp[0:L, i] = feature_1D_all_complete[:,i]
        feature_1D_all_complete = X_tmp
      X_1D[pdb_indx, :, :] = feature_1D_all_complete
      if len(feature_2D_all_complete[0, 0, :]) != F_2D:
        print('ERROR! 2D Feature length of ',sample_pdb,' not equal to ',pdb_name)
        exit;

      ### expand to lmax
      if feature_2D_all_complete.shape[0] < l_max:
        L = feature_2D_all_complete.shape[0]
        F = feature_2D_all_complete.shape[2]
        X_tmp = np.zeros((l_max, l_max, F))
        for i in range (0, F):
          X_tmp[0:L,0:L, i] = feature_2D_all_complete[:,:,i]
        feature_2D_all_complete = X_tmp
      X_2D[pdb_indx, :, :, :] = feature_2D_all_complete
      pdb_indx = pdb_indx + 1
  return (X_1D,X_2D)


def get_x_2D_from_this_list(selected_ids, feature_dir, l_max,dist_string):
  xcount = len(selected_ids)
  sample_pdb = ''
  for pdb in selected_ids:
    sample_pdb = pdb
    break
  featurefile =feature_dir + 'X-'  + sample_pdb + '.txt'
  print(featurefile)
  ### load the data
  (featuredata,feature_index_all_dict) = getX_2D_format(featurefile, reject_fea_file='None')     
  ### merge 1D data to L*m
  ### merge 2D data to  L*L*n
  feature_2D_all=[]
  for key in sorted(feature_index_all_dict.keys()):
      featurename = feature_index_all_dict[key]
      feature = featuredata[key]
      feature = np.asarray(feature)
      #print("keys: ", key, " featurename: ",featurename, " feature_shape:", feature.shape)
      
      if feature.shape[0] == feature.shape[1]:
        feature_2D_all.append(feature)
      else:
        print("Wrong dimension")
  
  fea_len = feature_2D_all[0].shape[0]
  F_2D = len(feature_2D_all)
  feature_2D_all = np.asarray(feature_2D_all)
  #print(feature_2D_all.shape)
  print("Total ",F_2D, " 2D features")
  X_2D = np.zeros((xcount, l_max, l_max, F_2D))
  pdb_indx = 0
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
      (featuredata,feature_index_all_dict) = getX_2D_format(featurefile, reject_fea_file='None')     
      ### merge 1D data to L*m
      ### merge 2D data to  L*L*n
      feature_2D_all=[]
      for key in sorted(feature_index_all_dict.keys()):
          featurename = feature_index_all_dict[key]
          feature = featuredata[key]
          feature = np.asarray(feature)
          #print("keys: ", key, " featurename: ",featurename, " feature_shape:", feature.shape)
          
          if feature.shape[0] == feature.shape[1]:
            feature_2D_all.append(feature)
          else:
            print("Wrong dimension")
     
      L = feature_2D_all[0].shape[0]
      F = len(feature_2D_all)
      X_tmp = np.zeros((L, L, F))
      for i in range (0, F):
        X_tmp[:,:, i] = feature_2D_all[i]      
      
      feature_2D_all = X_tmp
      #print feature_2D_all.shape #(123, 123, 18) 
      if len(feature_2D_all[0, 0, :]) != F_2D:
        print('ERROR! 2D Feature length of ',sample_pdb,' not equal to ',pdb_name)
        exit;

      ### expand to lmax
      if feature_2D_all[0].shape[0] < l_max:
        print("extend to lmax: ",feature_2D_all.shape)
        L = feature_2D_all.shape[0]
        F = feature_2D_all.shape[2]
        X_tmp = np.zeros((l_max, l_max, F))
        for i in range (0, F):
          X_tmp[0:L,0:L, i] = feature_2D_all[:,:,i]
        feature_2D_all_complete = X_tmp
      X_2D[pdb_indx, :, :, :] = feature_2D_all_complete
      pdb_indx = pdb_indx + 1
  return X_2D


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
  (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = print_detailed_evaluations(dict_l, dict_n, dict_e, P3L5, P3L2, P31L, Y)
  return (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l)

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
  return (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l)



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
  (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = print_detailed_evaluations(dict_l, dict_n, dict_e, P3L5, P3L2, P31L, Y)
  return (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l)