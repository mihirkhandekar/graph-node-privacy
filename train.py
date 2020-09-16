import itertools
import math
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.metrics as skmet
import stellargraph as sg
from sklearn import feature_extraction, model_selection, preprocessing, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from stellargraph import datasets, globalvar
from stellargraph.layer import GCN
from stellargraph.mapper import FullBatchNodeGenerator
from tensorflow.keras import Model, layers, losses, metrics, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from xgboost import XGBClassifier, XGBRegressor

from data import Data
from model import get_model, get_train_data, get_target_encoding
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

DATASET_NAME = "pubmed"
TRAIN_RATIO = 0.5
TRAIN = True

## Getting Graph Data
dataset = Data(DATASET_NAME)
node_data, node_label, edgelist, feature_names = dataset.get_data()
num_nodes = len(node_data.index)
num_edges = len(edgelist.index)
print('Dataset {} :: Nodes : {}, Edges : {}'.format(DATASET_NAME, num_nodes, num_edges))

## Splitting nodes into IN and OUT
node_ids = list(node_data.index) # List of all index of nodes
num_nodes_in = int(TRAIN_RATIO * num_nodes)
node_ids_in = list(random.sample(node_ids, num_nodes_in))
node_ids_out = np.setdiff1d(node_ids, node_ids_in)
print('IN nodes : {}, OUT nodes : {} (Total : {})'.format(len(node_ids_in), len(node_ids_out), num_nodes))
node_data_in = node_data.loc[node_ids_in, :]
node_data_out = node_data.loc[node_ids_out, :]


## Splitting edges into IN and OUT
edgelist_in = edgelist[0:0]
edgelist_out = edgelist[0:0]
for i, edge in edgelist.iterrows():
    if edge['target'] in node_ids_in or edge['source'] in node_ids_in:
        edgelist_in = edgelist_in.append(edgelist.iloc[i])
    else:
        edgelist_out = edgelist_out.append(edgelist.iloc[i])
print('IN edges : {}, OUT edges : {} (Total : {})'.format(len(edgelist_in.index), len(edgelist_out.index), num_edges))

## Training/Fetching GNN model
train_data, val_data, train_targets, val_targets = get_train_data(node_data_in, node_label)    # Split IN data into train and validation sets
model, train_gen, val_gen, generator = get_model(node_data[feature_names], edgelist_in, train_data.index, train_targets, val_data.index, val_targets)       
                                                                                                # Although all Node IDs are used, 
                                                                                                # only the IN edgelist is used for training

if TRAIN:
    history = model.fit_generator(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,
        callbacks=[ModelCheckpoint("logs/best_model.h5")]
    )
    test_gen = generator.flow(node_data_out.index, get_target_encoding(node_data_out, node_label))
    test_metrics = model.evaluate_generator(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
else:
    model.load_weights("logs/best_model.h5")

################# ATTACK 1 : Attacker knowledge of node features
print("Attack 1 : Attacker has knowledge of node features")

