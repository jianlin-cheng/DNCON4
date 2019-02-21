#!/usr/bin/env python

# Whole map approach

from __future__ import print_function

import sys
import os,glob 
sys.setrecursionlimit(1000000) 
#os.environ['THEANO_FLAGS'] = "device=cuda0"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time
import datetime

from math import sqrt

import numpy as np
import theano
import theano.tensor as T

import lasagne  ### require theano=0.7

from scipy.sparse import coo_matrix

from nndef import build_cnn_linear

MAP_CHANNELS = 60
RAW_CHANNELS = 441

RESTART_FLAG = 0
LRATE_DECAY = 0

# Locations of .map and .21c files (latter calculated from .aln files)
dc_train_map_dir = '/home/jh7x3/scratch/DNCON4/training/distance_classification/data/target_RealDistance/'
dc_train_21c_dir = '/home/jh7x3/scratch/DNCON4/training/distance_classification/data/aln21stats/'
model_dir = '/home/jh7x3/scratch/DNCON4/training/distance_classification/training/DeepCov_Res18/'

# ################## Download and prepare the dataset ##################

def load_dataset():

    length_list = []
    train_list = []
    validation_list = []
    ftmap_list = []
    targmap_list = []
    wtmap_list = []
    tnum = 0

    with open('/home/jh7x3/scratch/DNCON4/training/distance_classification/training/DeepCov_Res18/train.lst', 'r') as targetfile:
    #with open('/home/farhan/Desktop/Projects/DeepCov/Data/fake_train_small.lst', 'r') as targetfile:
        for line in targetfile:
            target = line.rstrip()
            #rawdata = np.memmap(dc_train_map_dir + target + '.map', dtype=np.float32, mode='r')#needs to be changed as label file
            rawdata = np.loadtxt(dc_train_map_dir + 'Y-realDist-' + target + '.txt', dtype=np.float32)#needs to be changed as label file, Y-realDist-1KMT-A.txt
            #length = int(sqrt(rawdata.shape[0]/(MAP_CHANNELS+1)))
            length = rawdata.shape[0] #(177, 177)
            length_list.append(length)
            #targmap = coo_matrix(rawdata.reshape(MAP_CHANNELS+1, length, length)[0,:,:])
            targmap = coo_matrix(rawdata.reshape(1, length, length)[0,:,:])
            targmap_list.append(targmap)

            ftmap_list.append(dc_train_21c_dir + target + '.21stats')

            #print(target,length)
            if (tnum < 130):
            #if (tnum < 40):
                validation_list.append(tnum)
            else:
                train_list.append(tnum)
            #train_list.append(tnum)
            #validation_list.append(tnum)
            tnum += 1
    print("Plotting length distribution of training list")
    upper =int(0.2*max(length_list)+0.1)
    lower =int(0.2*min(length_list)-0.1)
    
    for i in [x * 5 for x in range(lower,upper)]:
        print(i,'\t|' ,'*'*length_list.count(i))
    
    return ftmap_list, targmap_list, train_list, validation_list, length_list


# ############################## Main program ################################

def main(num_epochs=200):

    # Prepare Theano variables for inputs and targets
    input_var = T.ftensor4('inputs')
    target_var = T.ftensor4('targets')
    wtmap_var = T.ftensor4('wtmaps')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn_linear(input_var)

    # Load the dataset
    print("Loading data...")
    ftmap_list, targmap_list, train_list, validation_list, length_list = load_dataset()

    ntargets = len(length_list)
    ntrain = len(train_list)
    nvalidation = len(validation_list)
    nsamples = ntrain + nvalidation
    print("{} validation. {} in train.\n".format(nvalidation,ntrain))
    print("{} maps read. {} samples in total.\n".format(ntargets,nsamples))
    #sys.exit()

    # Create a loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = lasagne.objectives.aggregate(loss, weights=wtmap_var, mode='mean')
    weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    loss += 1e-4 * weightsl2

    # Variables for learning rate decay
    current_lr = theano.shared(np.array(0.000136, dtype=theano.config.floatX))
    lr_inc = np.array(1.1, dtype=theano.config.floatX)
    lr_dec = np.array(0.5, dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable=True)

    # Use adamax for first epoch to avoid extreme initial weight changes
    firstupdates = lasagne.updates.adamax(loss, params)
    # Then Nesterov momentum with learning rate decay for weight decay
    if LRATE_DECAY:
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=current_lr)
    else:
        updates = lasagne.updates.adamax(loss, params)

    # Validation/test functions
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = lasagne.objectives.aggregate(test_loss, weights=wtmap_var, mode='mean')
    # Maintain extra variables for MCC calculation in Theano
    #true_pos_0 = T.sum(T.ge(test_prediction, 0.5) * T.ge(target_var, 0.5) * T.gt(wtmap_var, 0.0), dtype=theano.config.floatX)
    #true_neg_0 = T.sum(T.lt(test_prediction, 0.5) * T.lt(target_var, 0.5) * T.gt(wtmap_var, 0.0), dtype=theano.config.floatX)
    #false_pos_0 = T.sum(T.ge(test_prediction, 0.5) * T.lt(target_var, 0.5) * T.gt(wtmap_var, 0.0), dtype=theano.config.floatX)
    #false_neg_0 = T.sum(T.lt(test_prediction, 0.5) * T.ge(target_var, 0.5) * T.gt(wtmap_var, 0.0), dtype=theano.config.floatX)

    # Compile functions to perform a training step
    train_func1 = theano.function([input_var, target_var, wtmap_var], loss, updates=firstupdates)
    train_func2 = theano.function([input_var, target_var, wtmap_var], loss, updates=updates)

    # Compile a second function computing the validation loss and TP/FP/FN/TN counts:
    #val_func = theano.function([input_var, target_var, wtmap_var], [test_loss, true_pos_0, true_neg_0, false_pos_0, false_neg_0])
    val_func = theano.function([input_var, target_var, wtmap_var], [test_loss])

    val_err_min = 1e32
    #val_mcc_max = -999
    #val_err_min = 1000000
    train_err_last = 1e32

    # Load current model snapshot
    #if RESTART_FLAG:
    #    with np.load('fullmap_metapsicov_model_regression.npz') as f:
    #        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    #    lasagne.layers.set_all_param_values(network, param_values)

    # Finally, launch the training loop.
    print("Starting training...")
    if os.path.exists(model_dir+'/weights/model.npz'): #'fullmap_metapsicov_model_best.npz'
        with np.load(model_dir+'/weights/model.npz') as f:
            print("Loading existing mdoel: ",model_dir,'/weights/model.npz',"\n",sep="")
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    else :
        print("Reinitializing weights")

    # Finally, launch the training loop. 
    indices = np.arange(ntrain)

    optcount = 0
    
    model_counter = len(glob.glob1(model_dir,"/weights/*.npz"))
    for epoch in range(model_counter-1,num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        sys.stdout.flush()

        np.random.shuffle(indices)
        for i in range(0, ntrain):
            tn = train_list[indices[i]]
            length = length_list[tn]
            #print(ftmap_list[tn])
            inputs = np.memmap(ftmap_list[tn], dtype=np.float32, mode='r', shape=(1,RAW_CHANNELS,length,length))

            targets = np.reshape(targmap_list[tn].toarray(), (1,1,length,length))

            wtmaps = np.ones((1,1,length,length), dtype=np.float32)
            rows, cols = np.indices((length,length))
            for ofs in range(-4, 5):
                row_vals = np.diag(rows, k=ofs)
                col_vals = np.diag(cols, k=ofs)
                wtmaps[0,0,row_vals,col_vals] = 0.0

            if epoch > 0:
                train_err += train_func2(inputs, targets, wtmaps)
            else:
                train_err += train_func1(inputs, targets, wtmaps)
            train_batches += 1
            #print((time.time() - start_time) * len(train_list) / (start_idx + BATCH_SIZE) / 3600.0)

        # And a full pass over the validation data:
        val_err = 0.0
        #val_tp = 0.0
        #val_tn = 0.0
        #val_fp = 0.0
        #val_fn = 0.0
        val_batches = 0
        for i in range(0, nvalidation):
            tn = validation_list[i]
            length = length_list[tn]
            #inputs = np.memmap(ftmap_list[tn], dtype=np.float32, mode='r', shape=(1,1,length*21,length*21))
            inputs = np.memmap(ftmap_list[tn], dtype=np.float32, mode='r', shape=(1,RAW_CHANNELS,length,length))

            targets = np.reshape(targmap_list[tn].toarray(), (1,1,length,length))

            wtmaps = np.ones((1,1,length,length), dtype=np.float32)
            rows, cols = np.indices((length,length))
            for ofs in range(-4, 5):
                row_vals = np.diag(rows, k=ofs)
                col_vals = np.diag(cols, k=ofs)
                wtmaps[0,0,row_vals,col_vals] = 0.0

            #err, tp, tn, fp, fn = val_func(inputs, targets, wtmaps)
            err = val_func(inputs, targets, wtmaps)[0]
            #print("Error: ",err)
            val_err += err
            #val_tp += tp
            #val_tn += tn
            #val_fp += fp
            #val_fn += fn
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        if (LRATE_DECAY and train_err_last != 1e32):
            if (train_err < train_err_last and current_lr.get_value() < 0.01):
                current_lr.set_value(current_lr.get_value() * lr_inc)
                print("Increasing learning rate to {} ...".format(current_lr.get_value()))
            else:
                current_lr.set_value(current_lr.get_value() * lr_dec)
                print("Decreasing learning rate to {} ...".format(current_lr.get_value()))
        train_err_last = train_err

        #val_mcc = sqrt((val_tp+val_fp)*(val_tp+val_fn)*(val_tn+val_fp)*(val_tn+val_fn))
    
        #if (val_mcc > 0.0):
        #    val_mcc = (val_tp * val_tn - val_fp * val_fn) / val_mcc

        #val_prec = val_tp / (val_tp + val_fp)
        #val_recall = val_tp / (val_tp + val_fn)
        #val_f1 = 2*((val_prec * val_recall)/(val_prec + val_recall))
        

        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        #print("  validation precision:\t\t{:.3f}".format(val_prec))
        #print("  validation recall:\t\t{:.3f}".format(val_recall))
        #print("  validation F1:\t\t{:.3f}".format(val_f1))
        #print("  validation MCC:\t\t{:.3f}".format(val_mcc))
        
        np.savez(model_dir+'/weights/model'+str(epoch)+'.npz', *lasagne.layers.get_all_param_values(network))

        if (val_err < val_err_min):
            val_err_min = val_err
            np.savez(model_dir+'/weights/fullmap_metapsicov_model_best.npz', *lasagne.layers.get_all_param_values(network))
            print("Best mse: ",val_err_min/val_batches," Saving model to fullmap_metapsicov_model_best.npz...")
            print("Saving model...")
            optcount = 0
        #else:
        #    optcount += 1
        #    if (optcount == 10):
        #        break
        print("Epoch {} took {:.3f}s".format(epoch, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        with open (model_dir+'/weights/training.loss',"a+") as f1:
                f1.write("{}: Epoch {} for train data got Training loss: {:.6f} and Validation loss: {:.6f}, took {:.3f}s\n".format(st, epoch, train_err / train_batches, val_err / val_batches, time.time() - start_time))
        
        print("Saving model to ",model_dir,'/model.npz',sep='')
        np.savez(model_dir+'/weights/model.npz', *lasagne.layers.get_all_param_values(network))
        ### release memory?
       #print("Checking interval memory")
       #os.system("nvidia-smi")
        

if __name__=="__main__":
    main()
