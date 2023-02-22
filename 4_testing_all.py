# tensorflow=2.4
import numpy as np
from pandas import read_csv
import tensorflow as tf
import argparse
import os
from models import GCNPolicy

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument("--dataTest", 		help="number of test data", 			default=1000)
parser.add_argument("--gpu", 			help="gpu index", 						default="0")
parser.add_argument("--embSize", 		help="embedding size of GNN", 			default=None)
parser.add_argument("--type", 			help="what's the type of the model", 	default="fea", 	choices = ['fea','obj','sol'])
parser.add_argument("--set", 			help="which set you want to test on?", 	default="train",choices = ['test','train'])
parser.add_argument("--loss", 			help="loss function used in testing", 	default="mse", 	choices = ['mse','l2'])
parser.add_argument("--folderModels", 	help="the folder of the saved models", 	default='./saved-models/')
args = parser.parse_args()

## FUNCTION OF TRAINING PER EPOCH
def process(model, dataloader, type = 'fea', loss = 'mse', n_Vars_small = 50):

	c, ei, ev, v, n_cs, n_vs, n_csm, n_vsm, cand_scores = dataloader
	batched_states = (c, ei, ev, v, n_cs, n_vs, n_csm, n_vsm)  
	logits = model(batched_states, tf.convert_to_tensor(False)) 
	
	return_err = None
	
	if type == "fea":
		errs_fp = np.sum((logits.numpy() > 0.5) & (cand_scores.numpy() < 0.5))
		errs_fn = np.sum((logits.numpy() < 0.5) & (cand_scores.numpy() > 0.5))
		errs = errs_fp + errs_fn
		return_err = errs / cand_scores.shape[0]
	
	if type == "obj":
		if loss == 'mse':
			loss = tf.keras.metrics.mean_squared_error(cand_scores, logits)
			return_err = tf.reduce_mean(loss).numpy()
		else:
			loss = ( tf.abs(cand_scores - logits) / (tf.abs(cand_scores) + 1.0) )
			return_err = tf.reduce_mean(loss).numpy()
			
	if type == "sol":
		if loss == 'mse':
			loss = tf.keras.metrics.mean_squared_error(cand_scores, logits)
			return_err = tf.reduce_mean(loss).numpy()
		else:
			length_sol = logits.shape[0]
			cand_scores = tf.reshape(cand_scores, [int(length_sol/n_Vars_small), n_Vars_small])
			logits = tf.reshape(logits, [int(length_sol/n_Vars_small), n_Vars_small])
			loss = tf.math.reduce_euclidean_norm(cand_scores - logits, axis = 1)
			norm = tf.math.reduce_euclidean_norm(cand_scores, axis = 1) + 1.0
			return_err = tf.reduce_mean(loss / norm).numpy()

	return return_err

## SET-UP DATASET
datafolder = "./data-training/" if args.set == "train" else "./data-testing/"
n_Samples_test = int(args.dataTest)
n_Cons_small = 10 # Each LP has 10 constraints
n_Vars_small = 50 # Each LP has 50 variables
n_Eles_small = 100 # Each LP has 100 nonzeros in matrix A

## SET-UP MODELS
model_list = []
for model_name in os.listdir(args.folderModels):
	model_path = args.folderModels + model_name
	model_type = model_name.split('_')[0]
	if model_type != args.type:
		continue
	n_Samples = int(model_name.split('_')[1][1:]) if args.set == "train" else n_Samples_test
	embSize = int(model_name.split('_')[2][1:-4])
	if args.embSize is not None and embSize != int(args.embSize):
		continue
	model_list.append((model_path, embSize, n_Samples))

## LOAD DATASET INTO MEMORY
if args.type == "fea":
	varFeatures_np = read_csv(datafolder + "VarFeatures_all.csv", header=None).values
	conFeatures_np = read_csv(datafolder + "ConFeatures_all.csv", header=None).values
	edgFeatures_np = read_csv(datafolder + "EdgeFeatures_all.csv", header=None).values
	edgIndices_np = read_csv(datafolder + "EdgeIndices_all.csv", header=None).values
	labels_np = read_csv(datafolder + "Labels_feas.csv", header=None).values
if args.type == "obj":
	varFeatures_np = read_csv(datafolder + "VarFeatures_feas.csv", header=None).values
	conFeatures_np = read_csv(datafolder + "ConFeatures_feas.csv", header=None).values
	edgFeatures_np = read_csv(datafolder + "EdgeFeatures_feas.csv", header=None).values
	edgIndices_np = read_csv(datafolder + "EdgeIndices_feas.csv", header=None).values
	labels_np = read_csv(datafolder + "Labels_obj.csv", header=None).values
if args.type == "sol":
	varFeatures_np = read_csv(datafolder + "VarFeatures_feas.csv", header=None).values
	conFeatures_np = read_csv(datafolder + "ConFeatures_feas.csv", header=None).values
	edgFeatures_np = read_csv(datafolder + "EdgeFeatures_feas.csv", header=None).values
	edgIndices_np = read_csv(datafolder + "EdgeIndices_feas.csv", header=None).values
	labels_np = read_csv(datafolder + "Labels_solu.csv", header=None).values

## SET-UP TENSORFLOW
gpu_index = int(args.gpu)
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

with tf.device("GPU:"+str(gpu_index)):

	for model_path, embSize, n_Samples in model_list:

		### LOAD DATASET INTO GPU ###
		varFeatures = tf.constant(varFeatures_np[:n_Vars_small * n_Samples,:], dtype=tf.float32)
		conFeatures = tf.constant(conFeatures_np[:n_Cons_small * n_Samples,:], dtype=tf.float32)
		edgFeatures = tf.constant(edgFeatures_np[:n_Eles_small * n_Samples,:], dtype=tf.float32)
		edgIndices = tf.constant(edgIndices_np[:n_Eles_small * n_Samples,:], dtype=tf.int32)
		edgIndices = tf.transpose(edgIndices)
		if args.type == "sol":
			labels = tf.constant(labels_np[:n_Vars_small * n_Samples,:], dtype=tf.float32)
		else:
			labels = tf.constant(labels_np[:n_Samples,:], dtype=tf.float32)
		nConsF = conFeatures.shape[1]
		nVarF = varFeatures.shape[1]
		nEdgeF = edgFeatures.shape[1]
		n_Cons = conFeatures.shape[0]
		n_Vars = varFeatures.shape[0]
		data = (conFeatures, edgIndices, edgFeatures, varFeatures, n_Cons, n_Vars, n_Cons_small, n_Vars_small, labels)

		### LOAD MODEL ###
		if args.type == "sol":
			model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF, isGraphLevel = False)
		else:
			model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF)
		model.restore_state(model_path)

		### TEST MODEL ###
		err = process(model, data, type = args.type, loss = args.loss, n_Vars_small = n_Vars_small)
		print(f"MODEL: {model_path}, DATA-SET: {datafolder}, NUM-DATA: {n_Samples}, LOSS: {args.loss}, ERR: {err}")
	
	

