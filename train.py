import os
import sys
import numpy as np
import theano
import theano.tensor as T
import time
import lasagne
import argparse
from metrics_mc import *
from model import neural_network
from confusionmatrix import ConfusionMatrix
from utils import iterate_minibatches

import h5py
import pandas as pd
from sklearn.metrics import average_precision_score


import sys
sys.dont_write_bytecode = True

import theano.sandbox.cuda.nvcc_compiler
print(theano.sandbox.cuda.nvcc_compiler.is_nvcc_available())

import pygpu
print(pygpu.test())


CATEGORIES_YAML = {"level1": [
							"actin-filaments", 
							"intermediate-filaments",
							"centrosome",
							"microtubules",
							"plasma-membrane",
							"cytosol",
							"lipid-droplets",
							"endoplasmic-reticulum",
							"golgi-apparatus",
							"vesicles",
							"endosomes",
							"lysosomes",
							"peroxisomes",
							"mitochondria",
							"nuclear-bodies",
							"nuclear-membrane",
							"nuclear-speckles",
							"nucleoli",
							"nucleoli-fibrillar-center",
							"nucleoplasm",
							"plastid"
							],

					"level2":[
							"cytoskeleton",
							"plasma-membrane",
							"endoplasmic-reticulum",
							"golgi-apparatus",
							"vesicles",
							"mitochondria",
							"nucleus",
							"nucleoili",
							"plastid",
							],

					"level3":[
							"cytoskeleton",
							"plasma-membrane",
							"cytosol",
							"endomembrane-system",
							"mitochondria",
							"nucleus",
							"nucleoli",
							"plastid"
							]
					}





def get_y(df, level, categories):
	one_hot = []
	for locs in df[f"level{level}"].str.split(";").to_list():
		temp = [1 if loc in locs else 0 for loc in categories]
		one_hot.append(temp)
	y = np.array(one_hot, dtype=np.float32)
	return y



parser = argparse.ArgumentParser()
parser.add_argument('-i', '--train_csv', default="data/uniprot_trainset.csv")
parser.add_argument('-t', '--test_csv',  default="data/hou_testset.csv")
parser.add_argument('-i_pssm', '--train_pssms',  default="data/DL1_uniprot.h5")
parser.add_argument('-t_pssm', '--test_pssms',  default="data/DL1_hou.h5")
parser.add_argument('-bs', '--batch_size',  help="Minibatch size, default = 128", default=128)
parser.add_argument('-e', '--epochs',  help="Number of training epochs, default = 200", default=200)
parser.add_argument('-n', '--n_filters',  help="Number of filters, default = 10", default=10)
parser.add_argument('-lr', '--learning_rate',  help="Learning rate, default = 0.0005", default=0.0005)
parser.add_argument('-id', '--in_dropout',  help="Input dropout, default = 0.2", default=0.2)
parser.add_argument('-hd', '--hid_dropout',  help="Hidden layers dropout, default = 0.5", default=0.5)
parser.add_argument('-hn', '--n_hid',  help="Number of hidden units, default = 256", default=256)
parser.add_argument('--savedir', type=str)
parser.add_argument('-m', '--multilabel', action='store_true', default=False)
parser.add_argument('-l', '--level', type=int, default=1)
parser.add_argument('-se', '--seed',  help="Seed for random number init., default = 123456", default=123456)
args = parser.parse_args()


# Input options
batch_size = int(args.batch_size)
n_hid = int(args.n_hid)
lr = float(args.learning_rate)
num_epochs = int(args.epochs)
drop_per = float(args.in_dropout)
drop_hid = float(args.hid_dropout)
n_filt = int(args.n_filters)
multilabel = args.multilabel
level = args.level
savedir = args.savedir

theano.config.floatX='float32'
lasagne.random.set_rng(np.random.RandomState(seed=int(args.seed)))
np.random.seed(seed=int(args.seed))

# Load data
print("Loading data...\n")
sys.stdout.flush()
categories = CATEGORIES_YAML[f"level{level}"]
n_class = len(categories)
trainset_pssms = h5py.File(args.train_pssms)
trainset = pd.read_csv(args.train_csv)
testset_pssms = h5py.File("data/DL1_hou.h5")
testset = pd.read_csv(args.test_csv)

#Train Data
train_ids = trainset.uniprot_id.to_numpy()
partition = trainset.fold.to_numpy()
num_folds = np.max(partition) + 1
X_train = []
for uid in train_ids:
	pssm = trainset_pssms[uid]
	X_train.append(pssm)
X_train = np.array(X_train) #(numsample, cliplen=1000, features=20)
seq_len = X_train.shape[1]
y_train = get_y(trainset, level, categories)
lens = trainset.Sequence.str.len().to_numpy()
lens = np.minimum(seq_len, lens)
a = np.tile(np.arange(seq_len),(lens.shape[0],1))
mask_train = np.array(a < lens[:,np.newaxis], dtype=np.int)

#Test Data
test_ids = testset.uniprot_id.to_numpy()
X_test = []
for uid in test_ids:
	pssm = testset_pssms[uid]
	X_test.append(pssm)
X_test = np.array(X_test)
y_test = get_y(testset, level, categories)
lens = testset.sequence.str.len().to_numpy()
lens = np.minimum(seq_len, lens)
a = np.tile(np.arange(seq_len),(lens.shape[0],1))
mask_test = np.array(a < lens[:,np.newaxis], dtype=np.int)


# Initialize utput vectors from test set
complete_alpha = np.zeros((X_test.shape[0],seq_len))
complete_context = np.zeros((X_test.shape[0],n_hid*2))
complete_test = np.zeros((X_test.shape[0],n_class))


# Number of features
n_feat = np.shape(X_test)[2]


best_val_loss = float('inf')
best_val_models = []

# Training
for i in range(num_folds):
	# Network compilation
	print("Compilation model {}\n".format(i))
	sys.stdout.flush()
	train_fn, val_fn, network_out = neural_network(batch_size, n_hid, n_feat, n_class, lr, drop_per, drop_hid, n_filt, multilabel=multilabel)
	
	# Train and validation sets
	train_index = np.where(partition != i)
	val_index = np.where(partition == i)
	X_tr = X_train[train_index].astype(np.float32)
	X_val = X_train[val_index].astype(np.float32)
	y_tr = y_train[train_index].astype(np.int32)
	y_val = y_train[val_index].astype(np.int32)
	mask_tr = mask_train[train_index].astype(np.float32)
	mask_val = mask_train[val_index].astype(np.float32)

	print("Validation shape: {}".format(X_val.shape))
	print("Training shape: {}".format(X_tr.shape))
	sys.stdout.flush()
	
	eps = []
	best_val_loss = 0

	print("Start training\n")
	sys.stdout.flush()	
	for epoch in range(num_epochs):
		# Calculate epoch time
		start_time = time.time()

		# Full pass training set
		train_err = 0
		train_batches = 0
		train_preds = []
		train_targets = []
	    
	    # Generate minibatches and train on each one of them	
		for batch in iterate_minibatches(X_tr, y_tr, mask_tr, batch_size, shuffle=True):
			inputs, targets, in_masks = batch
			targets = targets.astype(np.float32)
			tr_err, predict = train_fn(inputs, targets, in_masks)
			train_err += tr_err
			train_batches += 1
			train_preds.append(predict)
			train_targets.append(targets)
	    
		train_loss = train_err / train_batches
		train_preds = np.vstack(train_preds)
		train_targets = np.vstack(train_targets)
		train_ap_macro = average_precision_score(train_targets, train_preds, average='macro')
		train_ap_micro = average_precision_score(train_targets, train_preds, average='micro')    

		
		# Full pass validation set
		val_err = 0
		val_batches = 0
		val_preds = []
		val_targets = []
	    
	    # Generate minibatches and train on each one of them	
		for batch in iterate_minibatches(X_val, y_val, mask_val, batch_size):
			inputs, targets, in_masks = batch
			targets = targets.astype(np.float32)
			err, predict_val, alpha, context = val_fn(inputs, targets, in_masks)
			val_err += err
			val_batches += 1
			val_preds.append(predict_val)
			val_targets.append(targets)

		val_loss = val_err / val_batches
		val_preds = np.vstack(val_preds)
		val_targets = np.vstack(val_targets)
		val_ap_macro = average_precision_score(val_targets, val_preds, average='macro')
		val_ap_micro = average_precision_score(val_targets, val_preds, average='micro')
            

        # Save the best model based on validation loss
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_val_model = (train_fn, val_fn, network_out)
			best_preds = val_preds
	


		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_loss))
		print("  validation loss:\t\t{:.6f}".format(val_loss))
		print("  training macro accuracy:\t\t{:.2f} %".format(train_ap_macro))
		print("  training micro accuracy:\t\t{:.2f} %".format(train_ap_micro))
		print("  validation marco accuracy:\t\t{:.2f} %".format(val_ap_macro))
		print("  validation mirco accuracy:\t\t{:.2f} %".format(val_ap_micro))
		sys.stdout.flush()
	

	# Save the weights of the best model
	network_out = best_val_model[2]
	np.savez(f"{savedir}/models/level{level}/best_model_weights_fold_{i}.npz", *lasagne.layers.get_all_param_values(network_out))
	best_val_models.append(best_val_model)
	np.savez(
			f"{savedir}/outputs/level{level}/val_predictions_fold_{i}.npy", 
		  	preds=best_preds,
			targets=y_val)


# Initialize utput vectors from test set	
complete_alpha = np.zeros((X_test.shape[0],seq_len))
complete_context = np.zeros((X_test.shape[0],n_hid*2))
complete_test = np.zeros((X_test.shape[0],n_class))

for fold_idx, best_val_model in enumerate(best_val_models):
	train_fn, val_fn, network_out = best_val_model

	# Matrices to store all output information
	test_alpha = np.array([], dtype=np.float32).reshape(0,seq_len)
	test_context = np.array([], dtype=np.float32).reshape(0,n_hid*2)
	test_pred = np.array([], dtype=np.float32).reshape(0,n_class)

	for batch in iterate_minibatches(X_test, y_test, mask_test, batch_size, shuffle=False, sort_len=False):
		inputs, targets, in_masks = batch
		targets = targets.astype(np.float32)
		err, net_out, alpha, context = val_fn(inputs, targets, in_masks)
		
		last_alpha = alpha[:,-1:,:].reshape((batch_size, seq_len))
		test_alpha = np.concatenate((test_alpha, last_alpha), axis=0)
		test_context = np.concatenate((test_context, context), axis=0)
		test_pred = np.concatenate((test_pred, net_out),axis=0)	

	#TODO better save name
	np.savez(
			f"{savedir}/outputs/level{level}/test_predictions_fold_{fold_idx}.npy", 
		  	preds=test_pred,
			targets=y_test)

	test_ap_macro = average_precision_score(y_test, test_pred, average='macro')
	test_ap_micro = average_precision_score(y_test, test_pred, average='micro')

	print("  test macro average precision:\t\t{:.2f}".format(test_ap_macro))
	print("  test micro average precision:\t\t{:.2f}".format(test_ap_micro))


	# Output matrices test set are summed at the end of each training
	# Will be averaged over all folds
	# :X_test.shape[0] is to ensure that we dont get zeros at the end from
	# the last batch which might be smaller than batch_size
	complete_test += test_pred[:X_test.shape[0]]
	complete_context += test_context[:X_test.shape[0]]
	complete_alpha += test_alpha[:X_test.shape[0]]

# The test output from the 4 trainings is averaged
test_softmax = complete_test / num_folds
context_vectors = complete_context / num_folds
alpha_weight = complete_alpha / num_folds

# Final test average precision
test_ap_macro = average_precision_score(y_test, test_softmax, average='macro')
test_ap_micro = average_precision_score(y_test, test_softmax, average='micro')

print("FINAL TEST RESULTS")
print("  test macro average precision:\t\t{:.2f}".format(test_ap_macro))
print("  test micro average precision:\t\t{:.2f}".format(test_ap_micro))


