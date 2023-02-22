# tensorflow=2.4
import numpy as np
from pandas import read_csv
import tensorflow as tf
import argparse
from models import GCNPolicy

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument("--data", 		help="number of training data", 		default=1000)
parser.add_argument("--dataTest", 	help="number of test data", 			default=1000)
parser.add_argument("--gpu", 		help="gpu index", 						default="0")
parser.add_argument("--embSize", 	help="embedding size of GNN", 			default="16")
parser.add_argument("--type", 		help="what's the type of the model", 	default="fea", 		choices = ['fea','obj','sol'])
parser.add_argument("--set", 		help="which set you want to test on?", 	default="train", 	choices = ['test','train'])
parser.add_argument("--loss", 		help="loss function used in testing", 	default="mse", 		choices = ['mse','l2'])
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
			loss = tf.reduce_mean(loss / norm).numpy()

	return return_err

## SET-UP DATASET
datafolder = "./data-training/" if args.set == "train" else "./data-testing/"
n_Samples_test = int(args.dataTest)
n_Cons_small = 10 # Each LP has 10 constraints
n_Vars_small = 50 # Each LP has 50 variables
n_Eles_small = 100 # Each LP has 100 nonzeros in matrix A

## SET-UP MODEL
embSize = int(args.embSize)
n_Samples = int(args.data)
model_path = './saved-models/' + args.type + '_d' + str(n_Samples) + '_s' + str(embSize) + '.pkl'

## LOAD DATASET INTO MEMORY
if args.type == "fea":
	varFeatures = read_csv(datafolder + "VarFeatures_all.csv", header=None).values[:n_Vars_small * n_Samples_test,:]
	conFeatures = read_csv(datafolder + "ConFeatures_all.csv", header=None).values[:n_Cons_small * n_Samples_test,:]
	edgFeatures = read_csv(datafolder + "EdgeFeatures_all.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
	edgIndices = read_csv(datafolder + "EdgeIndices_all.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
	labels = read_csv(datafolder + "Labels_feas.csv", header=None).values[:n_Samples_test,:]
if args.type == "obj":
	varFeatures = read_csv(datafolder + "VarFeatures_feas.csv", header=None).values[:n_Vars_small * n_Samples_test,:]
	conFeatures = read_csv(datafolder + "ConFeatures_feas.csv", header=None).values[:n_Cons_small * n_Samples_test,:]
	edgFeatures = read_csv(datafolder + "EdgeFeatures_feas.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
	edgIndices = read_csv(datafolder + "EdgeIndices_feas.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
	labels = read_csv(datafolder + "Labels_obj.csv", header=None).values[:n_Samples_test,:]
if args.type == "sol":
	varFeatures = read_csv(datafolder + "VarFeatures_feas.csv", header=None).values[:n_Vars_small * n_Samples_test,:]
	conFeatures = read_csv(datafolder + "ConFeatures_feas.csv", header=None).values[:n_Cons_small * n_Samples_test,:]
	edgFeatures = read_csv(datafolder + "EdgeFeatures_feas.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
	edgIndices = read_csv(datafolder + "EdgeIndices_feas.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
	labels = read_csv(datafolder + "Labels_solu.csv", header=None).values[:n_Vars_small * n_Samples_test,:]

nConsF = conFeatures.shape[1]
nVarF = varFeatures.shape[1]
nEdgeF = edgFeatures.shape[1]
n_Cons = conFeatures.shape[0]
n_Vars = varFeatures.shape[0]

## SET-UP TENSORFLOW
gpu_index = int(args.gpu)
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

with tf.device("GPU:"+str(gpu_index)):

	### LOAD DATASET INTO GPU ###
	varFeatures = tf.constant(varFeatures, dtype=tf.float32)
	conFeatures = tf.constant(conFeatures, dtype=tf.float32)
	edgFeatures = tf.constant(edgFeatures, dtype=tf.float32)
	edgIndices = tf.constant(edgIndices, dtype=tf.int32)
	edgIndices = tf.transpose(edgIndices)
	labels = tf.constant(labels, dtype=tf.float32)
	data = (conFeatures, edgIndices, edgFeatures, varFeatures, n_Cons, n_Vars, n_Cons_small, n_Vars_small, labels)

	### LOAD MODEL ###
	if args.type == "sol":
		model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF, isGraphLevel = False)
	else:
		model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF)
	model.restore_state(model_path)

	### TEST MODEL ###
	err = process(model, data, type = args.type, loss = args.loss, n_Vars_small = n_Vars_small)
	model.summary()
	print(f"MODEL: {model_path}, DATA-SET: {datafolder}, NUM-DATA: {n_Samples_test}, LOSS: {args.loss}, ERR: {err}")
	
	

