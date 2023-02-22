# This script generates random LP instances for training and testing

import numpy as np
import scipy.optimize as opt
import random as rd
import os
import argparse
from pandas import read_csv


## ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--k_train",    default='2500')
parser.add_argument("--k_test",     default='1000')
parser.add_argument("--m",          default='10')
parser.add_argument("--n",          default='50')
parser.add_argument("--nnz",        default='100')
parser.add_argument("--prob",       default="0.3")
args = parser.parse_args()

## SETUP
k_data_training = int(args.k_train) # number of training data 
k_data_testing = int(args.k_test)   # number of testing data
m = int(args.m)                     # number of constraints
n = int(args.n)                     # number of variables
nnz = int(args.nnz)                 # number of nonzero elements in A 
prob_equal = float(args.prob)       # the probability that a constraint is a equality constraint
folder_training = "./data-training" # folder to save training data 
folder_testing  = "./data-testing"  # folder to save testing data


## DATA GENERATION
def generateLP(k_data, configs, folder):
    '''
    This function generates and saves LP instances.
    - k_data: the number of instances you want to generate 
    - configs: (m,n,nnz,prob_equal), configurations of each LP instance 
    - folder: the folder you want to save those generated LPs
    '''
    m,n,nnz,prob_equal = configs

    for k in range(k_data):
        path = folder + "/Data" + str(k)
        if not os.path.exists(path):
            os.makedirs(path)
        
        # randomly sample a LP problem
        # min c^T x
        # s.t. Aub x <= bub, Aeq x = beq, lb <= x <= ub
        c = np.random.uniform(-1, 1, n) * 0.01
        b = np.random.uniform(-1, 1, m)
        
        bounds = np.random.normal(0, 10, size = (n, 2))
        
        for j in range(n):
            if bounds[j, 0] > bounds[j, 1]:
                temp = bounds[j, 0]
                bounds[j, 0] = bounds[j, 1]
                bounds[j, 1] = temp
            
        A = np.zeros((m, n))
        EdgeIndex = np.zeros((nnz, 2))
        EdgeIndex1D = rd.sample(range(m * n), nnz)
        EdgeFeature = np.random.normal(0, 1, nnz)
        
        for l in range(nnz):
            i = int(EdgeIndex1D[l] / n)
            j = EdgeIndex1D[l] - i * n
            EdgeIndex[l, 0] = i
            EdgeIndex[l, 1] = j
            A[i, j] = EdgeFeature[l]
        
        circ = np.random.binomial(1, prob_equal, size = m)  # 1 means = constraint, 0 means <= constraint
        A_ub = A[circ == 0, :]
        b_ub = b[circ == 0]
        A_eq = A[circ == 1, :]
        b_eq = b[circ == 1]
        
        # solve the LP problem
        result = opt.linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = bounds)
        
        # write to CSV files
        np.savetxt(path + '/ConFeatures.csv', np.hstack((b.reshape(m, 1), circ.reshape(m, 1))), delimiter = ',', fmt = '%10.5f')
        np.savetxt(path + '/EdgeFeatures.csv', EdgeFeature, fmt = '%10.5f')
        np.savetxt(path + '/EdgeIndices.csv', EdgeIndex, delimiter = ',', fmt = '%d')
        np.savetxt(path + '/VarFeatures.csv', np.hstack((c.reshape(n, 1), bounds)), delimiter = ',', fmt = '%10.5f')
        np.savetxt(path + '/Labels_feas.csv', [result.status], fmt = '%d')
        
        if result.status != 2:  # feasible
            np.savetxt(path + '/Labels_obj.csv', [result.fun], fmt = '%10.5f')
            np.savetxt(path + '/Labels_solu.csv', result.x, fmt = '%10.5f')

        if k % 100 == 0:
            print('Generated:',k)


def combineGraphsAll(k_data, configs, folder):
    '''
    This function combines all LP instances in "folder" to a large graph to facilitate training.
    This function also makes labels for the feasibility of LP instances 
    '''
    m,n,nnz,prob_equal = configs

    ConFeatures_all = np.zeros((k_data * m, 2))
    EdgeFeatures_all = np.zeros((k_data * nnz, 1))
    EdgeIndices_all = np.zeros((k_data * nnz, 2))
    VarFeatures_all = np.zeros((k_data * n, 3))
    Labels_feas = np.zeros((k_data, 1))

    for k in range(k_data):
        LPfolder = folder + "/Data" + str(k)
        varFeatures = read_csv(LPfolder + "/VarFeatures.csv", header=None).values
        conFeatures = read_csv(LPfolder + "/ConFeatures.csv", header=None).values
        edgeFeatures = read_csv(LPfolder + "/EdgeFeatures.csv", header=None).values
        edgeIndices = read_csv(LPfolder + "/EdgeIndices.csv", header=None).values
        labelsFeas = read_csv(LPfolder + "/Labels_feas.csv", header=None).values
        
        edgeIndices[:, 0] = edgeIndices[:, 0] + k * m
        edgeIndices[:, 1] = edgeIndices[:, 1] + k * n
        
        ConFeatures_all[range(k * m, (k + 1) * m), :] = conFeatures
        VarFeatures_all[range(k * n, (k + 1) * n), :] = varFeatures
        EdgeFeatures_all[range(k * nnz, (k + 1) * nnz), :] = edgeFeatures
        EdgeIndices_all[range(k * nnz, (k + 1) * nnz), :] = edgeIndices
        Labels_feas[k] = 1 - labelsFeas / 2 # 0 means infeasible, 1 means feasible

        if k % 100 == 0:
            print("Combined:", k)
            
    np.savetxt(folder + '/ConFeatures_all.csv',     ConFeatures_all,    delimiter = ',', fmt = '%10.5f')
    np.savetxt(folder + '/EdgeFeatures_all.csv',    EdgeFeatures_all,   fmt = '%10.5f')
    np.savetxt(folder + '/EdgeIndices_all.csv',     EdgeIndices_all,    delimiter = ',', fmt = '%d')
    np.savetxt(folder + '/VarFeatures_all.csv',     VarFeatures_all,    delimiter = ',', fmt = '%10.5f')
    np.savetxt(folder + '/Labels_feas.csv',         Labels_feas,        fmt = '%10.5f')


def combineGraphsFeas(k_data, configs, folder):
    '''
    This function combines all feasible LP instances in "folder".
    This function also makes labels for the optimal objective and optimal solution.
    '''
    m,n,nnz,prob_equal = configs

    # collect info: which LP instances are feasible
    k_list = []
    k_feas = 0
    for k in range(k_data):
        LPfolder = folder + "/Data" + str(k)
        if os.path.exists(LPfolder + '/Labels_solu.csv'):
            k_list.append(k)
            k_feas = k_feas + 1

    ConFeatures_feas = np.zeros((k_feas * m, 2))
    EdgeFeatures_feas = np.zeros((k_feas * nnz, 1))
    EdgeIndices_feas = np.zeros((k_feas * nnz, 2))
    VarFeatures_feas = np.zeros((k_feas * n, 3))
    Labels_solu = np.zeros((k_feas * n, 1))
    Labels_obj = np.zeros((k_feas, 1))

    for l in range(k_feas):
        k = k_list[l]
        LPfolder = folder + "/Data" + str(k)
        varFeatures = read_csv(LPfolder + "/VarFeatures.csv", header=None).values
        conFeatures = read_csv(LPfolder + "/ConFeatures.csv", header=None).values
        edgeFeatures = read_csv(LPfolder + "/EdgeFeatures.csv", header=None).values
        edgeIndices = read_csv(LPfolder + "/EdgeIndices.csv", header=None).values
        labelsSolu = read_csv(LPfolder + "/Labels_solu.csv", header=None).values
        labelsObj = read_csv(LPfolder + "/Labels_obj.csv", header=None).values
        
        edgeIndices[:, 0] = edgeIndices[:, 0] + l * m
        edgeIndices[:, 1] = edgeIndices[:, 1] + l * n
        
        ConFeatures_feas[range(l * m, (l + 1) * m), :] = conFeatures
        VarFeatures_feas[range(l * n, (l + 1) * n), :] = varFeatures
        EdgeFeatures_feas[range(l * nnz, (l + 1) * nnz), :] = edgeFeatures
        EdgeIndices_feas[range(l * nnz, (l + 1) * nnz), :] = edgeIndices
        Labels_solu[range(l * n, (l + 1) * n), :] = labelsSolu
        Labels_obj[l] = labelsObj

        if l % 100 == 0:
            print("Combined:", l,'/',k_feas)
                   
    np.savetxt(folder + '/ConFeatures_feas.csv',    ConFeatures_feas,   delimiter = ',', fmt = '%10.5f')
    np.savetxt(folder + '/EdgeFeatures_feas.csv',   EdgeFeatures_feas,  fmt = '%10.5f')
    np.savetxt(folder + '/EdgeIndices_feas.csv',    EdgeIndices_feas,   delimiter = ',', fmt = '%d')
    np.savetxt(folder + '/VarFeatures_feas.csv',    VarFeatures_feas,   delimiter = ',', fmt = '%10.5f')
    np.savetxt(folder + '/Labels_solu.csv',         Labels_solu,        fmt = '%10.5f')
    np.savetxt(folder + '/Labels_obj.csv',          Labels_obj,         fmt = '%10.5f')


## MAIN SCRIPT
print("Generating training data.")
generateLP(k_data_training, (m,n,nnz,prob_equal), folder_training)
combineGraphsAll(k_data_training, (m,n,nnz,prob_equal), folder_training)
combineGraphsFeas(k_data_training, (m,n,nnz,prob_equal), folder_training)
print("Generating testing data.")
generateLP(k_data_testing, (m,n,nnz,prob_equal), folder_testing)
combineGraphsAll(k_data_testing, (m,n,nnz,prob_equal), folder_testing)
combineGraphsFeas(k_data_testing, (m,n,nnz,prob_equal), folder_testing)
