# On Representing Linear Programs by Graph Neural Networks


This repository is an implementation of the paper entitled "On Representing Linear Programs by Graph Neural Networks." (ICLR 2023 spotlight paper) The paper can be found [here](https://openreview.net/forum?id=cP2QVK-uygd). Our codes are modified from [this repo](https://github.com/ds4dm/learn2branch).

## Introduction

Learning to optimize is a rapidly growing area that aims to solve optimization problems or improve existing optimization algorithms using machine learning (ML). In particular, the graph neural network (GNN) is considered a suitable ML model for optimization problems whose variables and constraints are permutation--invariant, for example, the linear program (LP). While the literature has reported encouraging numerical results, this paper establishes the theoretical foundation of applying GNNs to solving LPs. Given any size limit of LPs, we construct a GNN that maps different LPs to different outputs. We show that properly built GNNs can reliably predict feasibility, boundedness, and an optimal solution for each LP in a broad class. Our proofs are based upon the recently--discovered connections between the Weisfeiler--Lehman isomorphism test and the GNN. To validate our results, we train a simple GNN and present its accuracy in mapping LPs to their feasibilities and solutions.

## A quick start guide

Step 1: Generating enough data
```
python 1_generate_data.py --k_train 2000 --k_test 1000 
# k_train and k_test means num. of training and testing LP instances
```
Step 2: Training and testing a GNN for the feasibility of LP.
```
python 2_training.py --type fea --data 500 --embSize 6
python 3_testing.py --type fea --set test --loss l2 --embSize 6 --data 500 --dataTest 500
```
Step 3: Training and testing a GNN for the objective of LP.
```
python 2_training.py --type obj --data 500 --embSize 6
python 3_testing.py --type obj --set test --loss l2 --embSize 6 --data 500 --dataTest 500
```
Step 4: Training and testing a GNN for the solution of LP.
```
python 2_training.py --type sol --data 500 --embSize 16
python 3_testing.py --type sol --set test --loss l2 --embSize 16 --data 500 --dataTest 500
```

## Reproducing all results

To reproduce all the results, please follow the commands in "cmds.txt"

Our environment: NVIDIA Tesla V100, CUDA 10.1, tensorflow 2.4.1.

## Related repo

On Representing Mixed-Integer Linear Programs by Graph Neural Networks:

https://github.com/liujl11git/GNN-MILP

## Citing our work

If you find our code helpful in your resarch or work, please cite our paper.
```
@inproceedings{
chen2023gnn-lp,
title={On Representing Linear Programs by Graph Neural Networks},
author={Ziang Chen and Jialin Liu and Xinshang Wang and Jianfeng Lu and Wotao Yin},
booktitle={International Conference on Learning Representations},
year={2023}
}
```

