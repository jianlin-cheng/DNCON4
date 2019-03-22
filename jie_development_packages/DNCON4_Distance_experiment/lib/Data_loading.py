# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:40:30 2017

@author: Jie Hou
"""
import os
import numpy as np
import math
import sys
import random
import keras.backend as K
epsilon = K.epsilon()

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

def chkfiles(fn):
  if os.path.exists(fn):
    return True 
  else:
    return False

# KeyError
# Feature file that has 0D, 1D, and 2D features (L is the first feature)
# Output size (a little >= L) to which all the features will be rolled up to as 2D features
def load_train_test_data_padding_with_interval(data_list, feature_dir,Interval,seq_end, min_seq_sep,dist_string, reject_fea_file='None'):
  import pickle
  
  #data_list ="/storage/htc/bdm/Collaboration/jh7x3/Contact_prediction_with_Tianqi/DNCON2_retrain_sort30/features_badri_db/DNCON2_retrain/badri_training_benchmark/lists-test-train/"
  #feature_dir='/storage/htc/bdm/Collaboration/jh7x3/Contact_prediction_with_Tianqi/DNCON2_retrain_sort30/features_badri_db/DNCON2_retrain/badri_training_benchmark/feats/'
  sequence_file=open(data_list,'r').readlines() 
  data_all_dict = dict()
  print("######### Loading data\n\t",end='')
  for i in range(0,len(sequence_file)):
      pdb_name = sequence_file[i].rstrip()
      print(pdb_name, "..",end='')
      featurefile = feature_dir + '/X-' + pdb_name + '.txt'
      if not os.path.isfile(featurefile):
                  print("feature file not exists: ",featurefile, " pass!")
                  continue
      cov = feature_dir + '/' + pdb_name + '.cov'
      if not os.path.isfile(cov):
                  print("Cov Matrix file not exists: ",cov, " pass!")
                  continue       
      plm = feature_dir + '/' + pdb_name + '.plm'
      if not os.path.isfile(plm):
                  print("plm matrix file not exists: ",plm, " pass!")
                  continue  
      targetfile = feature_dir + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
      if not os.path.isfile(targetfile):
                  print("target file not exists: ",targetfile, " pass!")
                  continue      
                                       
      
      ### load the data
      (featuredata,feature_index_all_dict) = getX_1D_2D(featurefile, cov, plm, reject_fea_file=reject_fea_file)
      
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
      #print("Checking length ",fea_len)
      for ran in range(0,seq_end,Interval):
          start_ran = ran
          end_ran = ran + Interval
          if end_ran > seq_end:
              end_ran = seq_end 
          if fea_len >start_ran and   fea_len <= end_ran:
              
              F_2D = len(feature_2D_all)
              
              X_2D = np.zeros((end_ran, end_ran, F_2D))
              for m in range (0, F_2D):
                X_2D[0:fea_len, 0:fea_len, m] = feature_2D_all[m]
                
              
              F_1D = len(feature_1D_all)
              
              X_1D = np.zeros((end_ran, F_1D))
              for m in range (0, F_1D):
                X_1D[0:fea_len, m] = feature_1D_all[m]
              
              
              ## load Y data          
              l_max = end_ran
              y = getY(targetfile, min_seq_sep, l_max)
              if (l_max * l_max != len(y)):
                print ('Error!! y does not have L * L feature values!!')
                sys.exit()
              
              fea_len_new=end_ran
              if fea_len_new in data_all_dict:
                  data_all_dict[fea_len_new].append([y,X_1D,X_2D])
              else:
                  data_all_dict[fea_len_new]=[]
                  data_all_dict[fea_len_new].append([y,X_1D,X_2D])             
          else:
              continue
  #for key in sorted(data_all_dict.keys()):
      #print("keys: ", key, " has ", len(data_all_dict), " samples",end='')
      #sample_list = data_all_dict[key]
      #for i in range(0,len(sample_list)):
      #  print("\t",i," 1D shape: ", sample_list[i][1].shape, "  2D shape:", sample_list[i][2].shape, "  y shape:", sample_list[i][0].shape)
  return data_all_dict
# train_datafile, feature_dir,inter,5000,0,dist_string, reject_fea_file
def load_train_test_data_padding_with_interval_2D(data_list, feature_dir,Interval,seq_end, min_seq_sep,dist_string, reject_fea_file='None', sample_flag = False):
  import pickle
  sequence_file=open(data_list,'r').readlines() 
  data_all_dict = dict()
  print("######### Loading data\n\t",end='')
  reject_list = []
  if reject_fea_file != 'None':
    # print("Loading ",reject_fea_file)
    with open(reject_fea_file) as f:
      for line in f:
        if line.startswith('-'):
          feature_name = line.strip()
          feature_name = feature_name[1:]
          # print("Removing ",feature_name)
          reject_list.append(feature_name)
  for i in range(0,len(sequence_file)):
      pdb_name = sequence_file[i].rstrip()
      print(pdb_name, "..",end='')
    
      featurefile = feature_dir + '/X-' + pdb_name + '.txt'
      if ('# cov' in reject_list or '# plm' in reject_list):
        if not os.path.isfile(featurefile):
                    print("feature file not exists: ",featurefile, " pass!")
                    continue     
      cov = feature_dir + '/' + pdb_name + '.cov'
      if '# cov' in reject_list:
        if not os.path.isfile(cov):
                    print("Cov Matrix file not exists: ",cov, " pass!")
                    continue        
      plm = feature_dir + '/' + pdb_name + '.plm'
      if '# plm' in reject_list:
        if not os.path.isfile(plm):
                    print("plm matrix file not exists: ",plm, " pass!")
                    continue       
      targetfile = feature_dir + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
      if not os.path.isfile(targetfile):
                  print("target file not exists: ",targetfile, " pass!")
                  continue                                 
      
      ### load the data
      (featuredata,feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, reject_list)
      
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
      #print("Checking length ",fea_len)
      for ran in range(0,seq_end,Interval):
          start_ran = ran
          end_ran = ran + Interval
          if end_ran > seq_end:
              end_ran = seq_end 
          if fea_len >start_ran and   fea_len <= end_ran:
              
              F = len(feature_2D_all) 
              X = np.zeros((end_ran, end_ran, F))
              for m in range (0, F):
                X[0:fea_len, 0:fea_len, m] = feature_2D_all[m]
              
              #print("Checking X of ", pdb_name)
              #print("Checking X shape ", X.reshape((1,X.shape[0],X.shape[1],X.shape[2])).shape)
              # print_feature_summary(X.reshape((1,X.shape[0],X.shape[1],X.shape[2])))
              ## load Y data          
              l_max = end_ran
              y = getY(targetfile, min_seq_sep, l_max)
              if (l_max * l_max != len(y)):
                print ('Error!! y does not have L * L feature values!!')
                sys.exit()
              
              fea_len_new=end_ran
              if fea_len_new in data_all_dict:
                  data_all_dict[fea_len_new].append([y,X])
              else:
                  data_all_dict[fea_len_new]=[]
                  data_all_dict[fea_len_new].append([y,X])             
          else:
              continue
  #for key in data_all_dict.keys():
      #print("keys: ", key, " has ", len(data_all_dict), " samples",end='')
      #sample_list = data_all_dict[key]
      #for i in range(0,len(sample_list)):
      #  print("\t",i, "  2D shape:", sample_list[i][1].shape, "  y shape:", sample_list[i][0].shape)
  return data_all_dict


def load_sample_data_2D(data_list, path_of_X, path_of_Y, Interval,seq_end, min_seq_sep,dist_string, reject_fea_file='None'):
  import pickle
  data_all_dict = dict()
  print("######### Loading data\n\t",end='')
  accept_list = []
  notxt_flag = True
  if reject_fea_file != 'None':
    with open(reject_fea_file) as f:
      for line in f:
        if line.startswith('#'):
          feature_name = line.strip()
          feature_name = feature_name[0:]
          accept_list.append(feature_name)
  ex_l = build_dataset_dictionaries_sample(data_list)
  sample_dict = subset_pdb_dict(ex_l, 0, 500, 5000, 'random') #can be random ordered
  sample_name = list(sample_dict.keys())
  sample_lens = list(sample_dict.values())
  for i in range(0,len(sample_name)):
    pdb_name = sample_name[i]
    pdb_lens = sample_lens[i]
    print(pdb_name, "..",end='')
    
    featurefile = path_of_X + '/X-' + pdb_name + '.txt'
    if ((len(accept_list) == 1 and ('# cov' not in accept_list and '# plm' not in accept_list)) or 
          (len(accept_list) == 2 and ('# cov' not in accept_list or '# plm' not in accept_list)) or (len(accept_list) > 2)):
      notxt_flag = False
      if not os.path.isfile(featurefile):
                  print("feature file not exists: ",featurefile, " pass!")
                  continue     
    cov = path_of_X + '/' + pdb_name + '.cov'
    if '# cov' in accept_list:
      if not os.path.isfile(cov):
                  print("Cov Matrix file not exists: ",cov, " pass!")
                  continue        
    plm = path_of_X + '/' + pdb_name + '.plm'
    if '# plm' in accept_list:
      if not os.path.isfile(plm):
                  print("plm matrix file not exists: ",plm, " pass!")
                  continue       
    targetfile = path_of_Y + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
    if not os.path.isfile(targetfile):
                print("target file not exists: ",targetfile, " pass!")
                continue                                 
      
    ### load the data
    (featuredata,feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, accept_list, pdb_lens, notxt_flag)
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
    #print("Checking length ",fea_len)
    for ran in range(0,seq_end,Interval):
        start_ran = ran
        end_ran = ran + Interval
        if end_ran > seq_end:
            end_ran = seq_end 
        if fea_len >start_ran and   fea_len <= end_ran:
            
            F = len(feature_2D_all) 
            X = np.zeros((end_ran, end_ran, F))
            for m in range (0, F):
              X[0:fea_len, 0:fea_len, m] = feature_2D_all[m]
                    
            l_max = end_ran
            y = getY(targetfile, min_seq_sep, l_max)
            if (l_max * l_max != len(y)):
              print ('Error!! y does not have L * L feature values!!')
              sys.exit()
            
            fea_len_new=end_ran
            if fea_len_new in data_all_dict:
                data_all_dict[fea_len_new].append([y,X])
            else:
                data_all_dict[fea_len_new]=[]
                data_all_dict[fea_len_new].append([y,X])             
        else:
            continue
  return data_all_dict

def print_feature_summary(X):
    print('FeatID         Avg        Med        Max        Sum        Avg[30]    Med[30]    Max[30]    Sum[30]')
    for ii in range(0, len(X[0, 0, 0, :])):
        (m,s,a,d) = (X[0, :, :, ii].flatten().max(), X[0, :, :, ii].flatten().sum(), X[0, :, :, ii].flatten().mean(), np.median(X[0, :, :, ii].flatten()))
        (m30,s30,a30, d30) = (X[0, 30, :, ii].flatten().max(), X[0, 30, :, ii].flatten().sum(), X[0, 30, :, ii].flatten().mean(), np.median(X[0, 30, :, ii].flatten()))
        print(' Feat%2s %10.4f %10.4f %10.4f %10.1f     %10.4f %10.4f %10.4f %10.4f' %(ii, a, d, m, s, a30, d30, m30, s30))



# Feature file that has 0D, 1D, and 2D features (L is the first feature)
# Output size (a little >= L) to which all the features will be rolled up to as 2D features
def getX_2D_format(feature_file, cov, plm, accept_list, pdb_len = 0, notxt_flag = True):
  # calcualte the length of the protein (the first feature)

  L = 0
  Data = []
  feature_all_dict = dict()
  feature_index_all_dict = dict() # to make sure the feature are same ordered 
  feature_name='None'
  feature_index=0
  # print(reject_list)
  if notxt_flag == True:
    L = pdb_len
  else:
    with open(feature_file) as f:
      for line in f:
        if line.startswith('#'):
          continue
        L = line.strip().split()
        L = int(round(math.exp(float(L[0]))))
        break
    with open(feature_file) as f:
      accept_flag = 1
      for line in f:
        if line.startswith('#'):
          if line.strip() not in accept_list:
            accept_flag = 0
          else:
            accept_flag = 1
          feature_name = line.strip()
          continue
        if accept_flag == 0:
          continue
        
        if line.startswith('#'):
          continue
        this_line = line.strip().split()
        if len(this_line) == 0:
          continue
        if len(this_line) == 1:
          # 0D feature
          continue
          # feature_namenew = feature_name + ' 0D'
          # feature_index +=1
          # if feature_index in feature_index_all_dict:
          #   print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          #   exit;
          # else:
          #   feature_index_all_dict[feature_index] = feature_namenew

          # feature0D = np.zeros((L, L))
          # feature0D[:, :] = float(this_line[0])
          # #feature0D = np.zeros((1, L))
          # #feature0D[0, :] = float(this_line[0])
          
          # if feature_index in feature_all_dict:
          #   print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          #   exit;
          # else:
          #   feature_all_dict[feature_index] = feature0D
        elif len(this_line) == L:
          # 1D feature
          continue
          # feature1D = np.zeros((1, L))

          # # 1D feature
          # feature1D1 = np.zeros((L, L))
          # feature1D2 = np.zeros((L, L))
          # for i in range (0, L):
          #   feature1D1[i, :] = float(this_line[i])
          #   feature1D2[:, i] = float(this_line[i])
          
          # ### load feature 1
          # feature_index +=1
          # feature_namenew = feature_name + ' 1D1'
          # if feature_index in feature_index_all_dict:
          #   print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          #   exit;
          # else:
          #   feature_index_all_dict[feature_index] = feature_namenew
          
          # if feature_index in feature_all_dict:
          #   print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          #   exit;
          # else:
          #   feature_all_dict[feature_index] = feature1D1
          
          # ### load feature 2
          # feature_index +=1
          # feature_namenew = feature_name + ' 1D2'
          # if feature_index in feature_index_all_dict:
          #   print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          #   exit;
          # else:
          #   feature_index_all_dict[feature_index] = feature_namenew

          # if feature_index in feature_all_dict:
          #   print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          #   exit;
          # else:
          #   feature_all_dict[feature_index] = feature1D2
        elif len(this_line) == L * L:
          # 2D feature
          feature2D = np.asarray(this_line).reshape(L, L)
          feature_index +=1
          feature_namenew = feature_name + ' 2D'
          if feature_index in feature_index_all_dict:
            print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
            exit
          else:
            feature_index_all_dict[feature_index] = feature_namenew
          
          if feature_index in feature_all_dict:
            print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
            exit
          else:
            feature_all_dict[feature_index] = feature2D
        else:
          print (line)
          print ('Error!! Unknown length of feature in !!' + feature_file)
          print ('Expected length 0, ' + str(L) + ', or ' + str (L*L) + ' - Found ' + str(len(this_line)))
          sys.exit()
####################Add Covariance Matrix #####################################
  if '# cov' in accept_list:   
      cov_rawdata = np.fromfile(cov, dtype=np.float32)
      length = int(math.sqrt(cov_rawdata.shape[0]/21/21))
      if length != L:
          print("Bad Alignment, pls check!")
          exit;
      inputs_cov = cov_rawdata.reshape(1,441,L,L)
      for i in range(441):
          feature2D = inputs_cov[0][i]
          feature_namenew = '# Covariance Matrix '+str(i+1)+ ' 2D'
          feature_index +=1
          if feature_index in feature_index_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              exit;
          else:
              feature_index_all_dict[feature_index] = feature_namenew
          if feature_index in feature_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              exit;
          else:
              feature_all_dict[feature_index] = feature2D
####################Add Pseudo_Likelihood Maximization #####################################
  if '# plm' in accept_list:  
      plm_rawdata = np.fromfile(plm, dtype=np.float32)
      length = int(math.sqrt(plm_rawdata.shape[0]/21/21))
      if length != L:
          print("Bad Alignment, pls check!")
          exit;
      inputs_plm = plm_rawdata.reshape(1,441,L,L)
      for i in range(441):
          feature2D = inputs_plm[0][i]
          feature_namenew = '# Pseudo_Likelihood Maximization '+str(i+1)+ ' 2D'
          feature_index +=1
          if feature_index in feature_index_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              exit;
          else:
              feature_index_all_dict[feature_index] = feature_namenew
          if feature_index in feature_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              exit;
          else:
              feature_all_dict[feature_index] = feature2D
  return (feature_all_dict,feature_index_all_dict)



# Feature file that has 0D, 1D, and 2D features (L is the first feature)
# Output size (a little >= L) to which all the features will be rolled up to as 2D features
def getX_1D_2D(feature_file, cov, plm, reject_fea_file='None'):
  # calcualte the length of the protein (the first feature)
  reject_list = []
  reject_list.append('# PSSM')
  reject_list.append('# AA composition')
  #print("Checking ",reject_fea_file)
  if reject_fea_file != 'None':
    #print("Loading ",reject_fea_file)
    with open(reject_fea_file) as f:
      for line in f:
        if line.startswith('-'):
          feature_name = line.strip()
          feature_name = feature_name[1:]
          #print("Removing ",feature_name)
          reject_list.append(feature_name)
  L = 0
  with open(feature_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      L = line.strip().split()
      L = int(round(math.exp(float(L[0]))))
      break
  Data = []
  feature_all_dict = dict()
  feature_index_all_dict = dict() # to make sure the feature are same ordered 
  feature_name='None'
  feature_index=0;
  with open(feature_file) as f:
    accept_flag = 1
    for line in f:
      if line.startswith('#'):
        if line.strip() in reject_list:
          accept_flag = 0
        else:
          accept_flag = 1
        feature_name = line.strip()
        continue
      if accept_flag == 0:
        continue
      
      if line.startswith('#'):
        continue
      this_line = line.strip().split()
      if len(this_line) == 0:
        continue
      if len(this_line) == 1:
        # 0D feature
        feature_namenew = feature_name + ' 0D'
        feature_index +=1
        if feature_index in feature_index_all_dict:
          print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          exit;
        else:
          feature_index_all_dict[feature_index] = feature_namenew
        
        feature0D = np.zeros((1, L))
        feature0D[0, :] = float(this_line[0])
        if feature_index in feature_all_dict:
          print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          exit;
        else:
          feature_all_dict[feature_index] = feature0D
      elif len(this_line) == L:
        # 1D feature
        feature1D = np.zeros((1, L))
        feature_namenew = feature_name + ' 1D'
        feature_index +=1
        if feature_index in feature_index_all_dict:
          print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          exit;
        else:
          feature_index_all_dict[feature_index] = feature_namenew
        
        for i in range (0, L):
          feature1D[0, i] = float(this_line[i])
        if feature_index in feature_all_dict:
          print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          exit;
        else:
          feature_all_dict[feature_index] = feature1D
      elif len(this_line) == L * L:
        # 2D feature
        feature2D = np.asarray(this_line).reshape(L, L)
        feature_namenew = feature_name + ' 2D'
        feature_index +=1
        if feature_index in feature_index_all_dict:
          print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          exit;
        else:
          feature_index_all_dict[feature_index] = feature_namenew
        if feature_index in feature_all_dict:
          print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          exit;
        else:
          feature_all_dict[feature_index] = feature2D
      else:
        print (line)
        print ('Error!! Unknown length of feature in !!' + feature_file)
        print ('Expected length 0, ' + str(L) + ', or ' + str (L*L) + ' - Found ' + str(len(this_line)))
        sys.exit()
  if '# cov' not in reject_list:
      cov_rawdata = np.fromfile(cov, dtype=np.float32)
      length = int(math.sqrt(cov_rawdata.shape[0]/21/21))
      if length != L:
          print("Bad Alignment, pls check!")
          exit;
      inputs_cov = cov_rawdata.reshape(1,441,L,L)
      for i in range(441):
          feature2D = inputs_cov[0][i]
          feature_namenew = '# Covariance Matrix '+str(i+1)+ ' 2D'
          feature_index +=1
          if feature_index in feature_index_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              exit;
          else:
              feature_index_all_dict[feature_index] = feature_namenew
          if feature_index in feature_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              exit;
          else:
              feature_all_dict[feature_index] = feature2D
  if '# plm' not in reject_list:
    plm_rawdata = np.fromfile(plm, dtype=np.float32)
    length = int(math.sqrt(plm_rawdata.shape[0]/21/21))
    if length != L:
        print("Bad Alignment, pls check!")
        exit;
    inputs_plm = plm_rawdata.reshape(1,441,L,L)
    for i in range(441):
        feature2D = inputs_plm[0][i]
        feature_namenew = '# Pseudo_Likelihood Maximization '+str(i+1)+ ' 2D'
        feature_index +=1
        if feature_index in feature_index_all_dict:
            print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
            exit;
        else:
            feature_index_all_dict[feature_index] = feature_namenew
        if feature_index in feature_all_dict:
            print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
            exit;
        else:
            feature_all_dict[feature_index] = feature2D
  return (feature_all_dict,feature_index_all_dict)


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


"""
source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/Keras1.2_TF1.5/bin/activate
python
featurefile = '/storage/htc/bdm/Collaboration/jh7x3/Contact_prediction_with_Tianqi/DNCON2_retrain_sort30/features_badri_db/DNCON2_retrain/badri_training_benchmark/feats/X-1B9O-A.txt'
featuredata = getX_1D_2D(featurefile, reject_fea_file='None')
featuredata.keys()

feature_1D_all=[]
feature_2D_all=[]
for key in featuredata.keys():
    myarray = np.asarray(featuredata[key])
    featuredata[key] = myarray.reshape(len(myarray),myarray.shape[1],myarray.shape[2])
    print "keys: ", key, ": ", featuredata[key].shape 
    if featuredata[key].shape[1] == featuredata[key].shape[2]:
      feature_2D_all.append(featuredata[key])
    else:
      feature_1D_all.append(featuredata[key])

for i in range(0,len(feature_1D_all)):
    print i, ": ", feature_1D_all[i].shape

feature_1D_all_tmp = np.concatenate(feature_1D_all)
feature_2D_all_tmp = np.concatenate(feature_2D_all)
feature_1D_all_tmp.shape  # (22, 1, 123)
feature_2D_all_tmp.shape # (18, 123, 123)
feature_1D_all_complete =  feature_1D_all_tmp.reshape(feature_1D_all_tmp.shape[0],feature_1D_all_tmp.shape[2]).transpose()
feature_1D_all_complete.shape #(123, 22)
feature_2D_all_complete =  feature_2D_all_tmp.reshape(feature_2D_all_tmp.shape[1],feature_2D_all_tmp.shape[2],feature_2D_all_tmp.shape[0])
feature_2D_all_complete.shape #(123, 123, 18)



# set "image_data_format": "channels_last" in ~/.keras/keras.json

#keys:  # Sequence separation between 38 and 48 2D :  (1, 123, 123)
#keys:  # Solvent accessibility 1D :  (1, 1, 123)
#keys:  # pref score 2D :  (1, 123, 123)
#keys:  # ccmpred 2D :  (1, 123, 123)
#keys:  # PSSM inf feature 1D :  (1, 1, 123)
#keys:  # PSSM sum cosines 2D :  (1, 123, 123)
#keys:  # levitt con pot 2D :  (1, 123, 123)
#keys:  # Sequence separation between 28 and 38 2D :  (1, 123, 123)
#keys:  # Relative 'E' count 0D :  (1, 1, 123)
#keys:  # freecontact 2D :  (1, 123, 123)
#keys:  # pstat_mimt 2D :  (1, 123, 123)
#keys:  # psicov 2D :  (1, 123, 123)
#keys:  # joint entro 2D :  (1, 123, 123)
#keys:  # Relative 'e' count 0D :  (1, 1, 123)
#keys:  # Sequence separation between 23 and 28 2D :  (1, 123, 123)
#keys:  # Relative 'H' count 0D :  (1, 1, 123)
#keys:  # braun con pot 2D :  (1, 123, 123)
#keys:  # pearson r 2D :  (1, 123, 123)
#keys:  # pstat_pots 2D :  (1, 123, 123)
#keys:  # Secondary Structure 1D :  (3, 1, 123)
#keys:  # Atchley factors 1D :  (5, 1, 123)
#keys:  # Sequence separation 48+ 2D :  (1, 123, 123)
#keys:  # Relative sequence separation 2D :  (1, 123, 123)
#keys:  # alignment-count (log) 0D :  (1, 1, 123)
#keys:  # Psipred 1D :  (3, 1, 123)
#keys:  # Sequence Length (log) 0D :  (1, 1, 123)
#keys:  # pstat_mip 2D :  (1, 123, 123)
#keys:  # PSSM Sums (divided by 100) 1D :  (1, 1, 123)
#keys:  # Psisolv 1D :  (1, 1, 123)
#keys:  # effective-alignment-count (log) 0D :  (1, 1, 123)
#keys:  # scld lu con pot 2D :  (1, 123, 123)
#keys:  # Shannon entropy sum 1D :  (1, 1, 123)


data_list = '/storage/htc/bdm/Collaboration/jh7x3/Contact_prediction_with_Tianqi/DNCON2_retrain_sort30/features_badri_db/DNCON2_retrain/badri_training_benchmark/lists-test-train/test.lst'
feature_dir = '/storage/htc/bdm/Collaboration/jh7x3/Contact_prediction_with_Tianqi/DNCON2_retrain_sort30/features_badri_db/DNCON2_retrain/badri_training_benchmark/feats/'
Interval=20
seq_end=1000
data_all_dict_padding = load_train_test_data_padding_with_interval(data_list, feature_dir,Interval,seq_end,24,10)



GLOBAL_PATH= '/storage/htc/bdm/jh7x3/DNCON4/architecture/CNN_arch'
print GLOBAL_PATH
sys.path.insert(0, GLOBAL_PATH+'/lib/')
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
DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt)
key=60
for key in data_all_dict_padding.keys():
    if key <start: # run first model on 100 at most
        continue
    if key > end: # run first model on 100 at most
        continue
    print '### Loading sequence length :', key

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


if seq_len in testdata_all_dict_padding:
    testfeaturedata = testdata_all_dict_padding[seq_len]
    #print "Loading test dataset "
else:
    testfeaturedata = trainfeaturedata
    print "\n\n##Warning: Setting training dataset as testing dataset \n\n"

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
train_2D_fea_all_array = train_2D_fea_all_array.reshape(train_2D_fea_all_array.shape[0],train_2D_fea_all_array.shape[1]*train_2D_fea_all_array.shape[2],train_2D_fea_all_array.shape[3])     #(21, 220*220, 18)  
  
test_label_all_array = np.asarray(test_label_all) #(21, 48400)
test_1D_fea_all_array = np.asarray(test_1D_fea_all) #(21, 220, 22)
test_2D_fea_all_array = np.asarray(test_2D_fea_all) #(21, 220, 220, 18)        
test_2D_fea_all_array = test_2D_fea_all_array.reshape(test_2D_fea_all_array.shape[0],test_2D_fea_all_array.shape[1]*test_2D_fea_all_array.shape[2],test_2D_fea_all_array.shape[3])  #(21, 220*220, 18)  

print "Train 1D shape: ",train_1D_fea_all_array.shape, " in outside epoch ", epoch 
print "Train 2D shape: ",train_2D_fea_all_array.shape, " in outside epoch ", epoch 
print "Test 1D shape: ",test_1D_fea_all_array.shape, " in outside epoch ", epoch
print "Test 2D shape: ",test_2D_fea_all_array.shape, " in outside epoch ", epoch

### Define the model 
model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)

sequence_length = seq_len

print "######## Setting initial model based on length ",sequence_length;
## ktop_node is the length of input proteins
if model_prefix == 'DNCON4_1d2dconv':
    # opt = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    DNCON4_CNN = DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,nb_filters,nb_layers,opt)

if os.path.exists(model_weight_out):
    print "######## Loading existing weights ",model_weight_out;
    DNCON4_CNN.load_weights(model_weight_out)
    DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
else:
    print "######## Setting initial weights";
    DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)


DNCON4_CNN.fit([train_1D_fea_all_array,train_2D_fea_all_array], train_label_all_array, batch_size= batch_size_train, nb_epoch=epoch_inside,  validation_data=([test_1D_fea_all_array,test_2D_fea_all_array], test_label_all_array), verbose=1)

DNCON4_CNN_prediction = DNCON4_CNN.predict([train_1D_fea_all_array,train_2D_fea_all_array])

DNCON4_CNN_prediction.shape
P = DNCON4_CNN_prediction
min_seq_sep=24
P2 = floor_lower_left_to_zero(P, min_seq_sep)
Y=train_label_all_array

path_lists='/storage/htc/bdm/Collaboration/jh7x3/Contact_prediction_with_Tianqi/DNCON2_retrain_sort30/features_badri_db/DNCON2_retrain/badri_training_benchmark/lists-test-train/'
tr_l, tr_n, tr_e, te_l, te_n, te_e = build_dataset_dictionaries(path_lists)
# Make combined dictionaries as well
all_l = te_l.copy()
all_n = te_n.copy()
all_e = te_e.copy()
all_l.update(tr_l)
all_n.update(tr_n)
all_e.update(tr_e)
print 'Total Number of Training and Test dataset = ' + str(len(all_l))


from libtrain import *

(list_acc_l5, list_acc_l2, list_acc_1l) = evaluate_prediction(LRT1, all_n, all_e, P, YRT1, 24)

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

"""
