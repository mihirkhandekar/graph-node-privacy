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
from scipy.special import entr
from sklearn import feature_extraction, model_selection, preprocessing, svm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from stellargraph import datasets, globalvar
from stellargraph.layer import GCN
from stellargraph.mapper import FullBatchNodeGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from xgboost import XGBClassifier, XGBRegressor

from data import Data
from model import GCNModel, get_target_encoding, get_train_data

SEED = 10
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

DATASET_NAME = "cora" #"financial"
EPOCHS = 20 #175
IN_RATIO = 0.1 #0.01
dropout = 0.
layer_sizes = [32, 32] #[32]
activations = ['tanh', 'tanh']

SHADOW_DATASET_NAME = "citeseer"
SHADOW_EPOCHS = 75
dropout_shadow = 0.0
layer_sizes_shadow = [8, ]
activations_shadow = ['tanh']

TRAIN = True

# Getting Graph Data
dataset = Data(DATASET_NAME)
node_data, node_label, edgelist, feature_names = dataset.get_data()
num_nodes = len(node_data.index)
num_edges = len(edgelist.index)
print('Dataset {} :: Nodes : {}, Edges : {}'.format(
    DATASET_NAME, num_nodes, num_edges))

# Splitting nodes into IN and OUT
node_ids = list(node_data.index)  # List of all index of nodes
num_nodes_in = int(IN_RATIO * num_nodes)
node_ids_in = list(random.sample(node_ids, num_nodes_in))
#node_ids_out = list(set(np.setdiff1d(node_ids, node_ids_in)) & set(node_data.index))
node_ids_out = np.setdiff1d(node_ids, node_ids_in)
print('IN nodes : {}, OUT nodes : {} (Total : {})'.format(
    len(node_ids_in), len(node_ids_out), num_nodes))
node_data_in = node_data.loc[node_ids_in, :]
node_data_out = node_data.loc[node_ids_out, :]


# Splitting edges into IN and OUT
edgelist_in = edgelist[edgelist['target'].isin(
    node_ids_in) | edgelist['source'].isin(node_ids_in)]

print('IN edges : {} (Total : {})'.format(
    len(edgelist_in.index), num_edges))

# Training/Fetching GNN model
train_data, val_data, train_targets, val_targets = get_train_data(
    node_data_in, node_label)    # Split IN data into train and validation sets
model = GCNModel(dropout=dropout, layer_sizes=layer_sizes,
                 activations=activations)
model, train_gen, val_gen, generator = model.get_model(
    node_data[feature_names], edgelist_in, train_data.index, train_targets, val_data.index, val_targets)
# Although all Node IDs are used,
# only the IN edgelist is used for training

if TRAIN:
    history = model.fit_generator(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,
        callbacks=[ModelCheckpoint("logs/best_model.h5")]
    )
    target_encoding = get_target_encoding(node_data_out, node_label)
    test_gen = generator.flow(node_data_out.index, target_encoding)

    test_metrics = model.evaluate_generator(test_gen)
    print("\nOUT Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    train_metrics = model.evaluate_generator(train_gen)
    print("\nIN Set Metrics:")
    for name, val in zip(model.metrics_names, train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

else:
    model.load_weights("logs/best_model.h5")

# ATTACK 1a : Attacker knowledge of node features
ATTACKER_NODE_KNOWLEDGE = 0.5
print("Attack 1 : Attacker has knowledge of node features")
num_nodes_in_know = int(len(node_ids_in) * ATTACKER_NODE_KNOWLEDGE)
num_nodes_out_know = int(len(node_ids_out) * ATTACKER_NODE_KNOWLEDGE)

min_know = min(num_nodes_in_know, num_nodes_out_know)
node_ids_in_know = random.sample(list(node_ids_in), min_know)
node_ids_out_know = random.sample(list(node_ids_out), min_know)
membership = [1 for i in range(len(node_ids_in_know))]
membership.extend([0 for i in range(len(node_ids_out_know))])

node_ids_know = list(node_ids_in_know)
node_ids_know.extend(node_ids_out_know)

node_data_know = node_data.loc[node_ids_know, :]
print('Attacker knowledge :: Nodes : {} {} (Should be equal)'.format(
    len(node_data_know.index), len(membership)))

edgelist_noedge = edgelist[0:0]
G_know = sg.StellarGraph(nodes={"paper": node_data[feature_names]}, edges={
    "cites": edgelist_noedge})       # node_data_know[feature_names] not used since does not matter here
generator_noedge = FullBatchNodeGenerator(G_know, method="gcn")
target_encoding = get_target_encoding(
    node_data_know, node_label)       # To see later

test_gen_noedge = generator_noedge.flow(node_data_know.index, target_encoding)
y = model.predict(test_gen_noedge)
squared = np.square(y[0] - target_encoding)
rms_loss = np.sqrt(np.sum(squared, axis=1) * (1/np.array(y).shape[1]))
entropies = entr(y).sum(axis=-1)/np.log(y.shape[1])

X = np.array(entropies[0]).T
kmeans = KMeans(n_clusters=2, max_iter=500, n_init=15).fit(X.reshape(-1, 1))
y = np.array(kmeans.labels_)
cluster_centers = kmeans.cluster_centers_
if cluster_centers[0][0] < cluster_centers[1][0]:
    y = -(y - 1)
correct = (y == np.array(membership))
print('#### Attack 1a Accuracy : (Node features) {} '.format(np.sum(correct)/len(correct)))

# Attack 1b
# , np.array(entropies[0]).reshape(-1, 1)
X = np.concatenate([np.array(rms_loss).reshape(-1, 1)], axis=1)
kmeans = KMeans(n_clusters=2, max_iter=500, n_init=15).fit(X)
y = np.array(kmeans.labels_)
cluster_centers = kmeans.cluster_centers_
if cluster_centers[0][0] < cluster_centers[1][0]:
    y = -(y - 1)
assert len(y) == len(membership)
correct = (y == np.array(membership))
print('#### Attack 1b Accuracy : (Node features + labels) {} '.format(np.sum(correct)/len(correct)))

# 1c
edgelist_know = edgelist[edgelist['target'].isin(
    node_ids_know) | edgelist['source'].isin(node_ids_know)]

print('ATTACK 1c :: Edges known : {}'.format(len(edgelist_know.index)))
G_know = sg.StellarGraph(nodes={"paper": node_data[feature_names]}, edges={
    "cites": edgelist_know})       # node_data_know[feature_names] not used since does not matter here
generator_noedge = FullBatchNodeGenerator(G_know, method="gcn")
target_encoding = get_target_encoding(
    node_data_know, node_label)       # To see later
test_gen_noedge = generator_noedge.flow(node_data_know.index, target_encoding)
y = model.predict(test_gen_noedge)
squared = np.square(y[0] - target_encoding)
rms_loss = np.sqrt(np.sum(squared, axis=1) * (1/np.array(y).shape[1]))
entropies = entr(y).sum(axis=-1)/np.log(y.shape[1])

X = np.array(entropies[0]).T
kmeans = KMeans(n_clusters=2, max_iter=500, n_init=15).fit(X.reshape(-1, 1))
y = np.array(kmeans.labels_)
cluster_centers = kmeans.cluster_centers_
if cluster_centers[0][0] < cluster_centers[1][0]:
    y = -(y - 1)
correct = (y == np.array(membership))
print('#### Attack 1c Accuracy : (Node features + edges) {} '.format(np.sum(correct)/len(correct)))


# Attack 2 : Shadow graph
shadow_dataset = Data(SHADOW_DATASET_NAME)
shadow_node_data, shadow_node_label, shadow_edgelist, shadow_feature_names = shadow_dataset.get_data()

shadow_num_nodes = len(shadow_node_data.index)
shadow_num_edges = len(shadow_edgelist.index)
print('Shadow Dataset {} :: Nodes : {}, Edges : {}'.format(
    SHADOW_DATASET_NAME, shadow_num_nodes, shadow_num_edges))

# Splitting nodes into IN and OUT
shadow_node_ids = list(shadow_node_data.index)  # List of all index of nodes
shadow_num_nodes_in = int(0.5 * shadow_num_nodes)
shadow_node_ids_in = list(random.sample(shadow_node_ids, shadow_num_nodes_in))
shadow_node_ids_out = np.setdiff1d(shadow_node_ids, shadow_node_ids_in)
print('SHADOW IN nodes : {}, OUT nodes : {} (Total : {})'.format(
    len(shadow_node_ids_in), len(shadow_node_ids_out), shadow_num_nodes))
shadow_node_data_in = shadow_node_data.loc[shadow_node_ids_in, :]
shadow_node_data_out = shadow_node_data.loc[shadow_node_ids_out, :]

# Splitting edges into IN and OUT
shadow_edgelist_in = shadow_edgelist[shadow_edgelist['target'].isin(
    node_ids_in) | shadow_edgelist['source'].isin(node_ids_in)]

print('SHADOW IN edges : {}, (Total : {})'.format(
    len(shadow_edgelist_in.index), shadow_num_edges))

# Training/Fetching GNN model
shadow_train_data, shadow_val_data, shadow_train_targets, shadow_val_targets = get_train_data(
    shadow_node_data_in, shadow_node_label)    # Split IN data into train and validation sets

shadow_model = GCNModel(dropout=dropout_shadow,
                        layer_sizes=layer_sizes_shadow, activations=activations_shadow)

shadow_model, shadow_train_gen, shadow_val_gen, shadow_generator = shadow_model.get_model(shadow_node_data[shadow_feature_names],
                                                                                          shadow_edgelist_in, shadow_train_data.index,
                                                                                          shadow_train_targets, shadow_val_data.index,
                                                                                          shadow_val_targets)
# Although all Node IDs are used,
# only the IN edgelist is used for training
history = shadow_model.fit_generator(
    shadow_train_gen,
    epochs=SHADOW_EPOCHS,
    validation_data=shadow_val_gen,
    verbose=2,
    shuffle=False,
    callbacks=[ModelCheckpoint("logs/best_model.h5")]
)
shadow_test_encoding = get_target_encoding(
    shadow_node_data_out, shadow_node_label)
shadow_train_encoding = get_target_encoding(
    shadow_node_data_in, shadow_node_label)
shadow_test_gen = shadow_generator.flow(
    shadow_node_data_out.index, shadow_test_encoding)
shadow_train_gen = shadow_generator.flow(
    shadow_node_data_in.index, shadow_train_encoding)

test_metrics = shadow_model.evaluate_generator(shadow_test_gen)
print("\nShadow OUT Set Metrics:")
for name, val in zip(shadow_model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
train_metrics = shadow_model.evaluate_generator(shadow_train_gen)
print("\nShadow IN Set Metrics:")
for name, val in zip(shadow_model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

in_preds = shadow_model.predict(shadow_train_gen)
out_preds = shadow_model.predict(shadow_test_gen)
in_entropies = np.array(entr(in_preds).sum(
    axis=-1)/np.log(in_preds.shape[1]))[0]
out_entropies = np.array(entr(out_preds).sum(
    axis=-1)/np.log(out_preds.shape[1]))[0]

shadow_entropies = list(in_entropies)
shadow_entropies.extend(out_entropies)
shadow_entropies = np.array(shadow_entropies)
shadow_membership = [1. for _ in range(len(in_entropies))]
shadow_membership.extend([0. for _ in range(len(out_entropies))])
shadow_membership = np.array(shadow_membership)

x_train, x_test, y_train, y_test = train_test_split(
    shadow_entropies.reshape(-1, 1), shadow_membership, test_size=0.2)

clf = svm.SVR()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("SVM Accuracy (Attack model) : %.2f%%" %
      (100.0 * accuracy_score(y_test, y_pred > 0.5)))

entropies = np.array(entropies[0]).reshape(-1, 1)
membership = np.array(membership)
y_pred = clf.predict(entropies)
print("#### Attack 2 Accuracy : : %.2f%%" %
      (100.0 * accuracy_score(membership, y_pred > 0.5)))


################# Attack 3
ATTACKER_NODE_KNOWLEDGE = 0.5
print("Attack 3 : Attacker has knowledge of subgraph")
num_nodes_in_know = int(len(node_ids_in) * ATTACKER_NODE_KNOWLEDGE)
num_nodes_out_know = int(len(node_ids_out) * ATTACKER_NODE_KNOWLEDGE)

min_know = min(num_nodes_in_know, num_nodes_out_know)
node_ids_in_know = random.sample(list(node_ids_in), min_know)
node_ids_out_know = random.sample(list(node_ids_out), min_know)
membership = [1 for i in range(len(node_ids_in_know))]
membership.extend([0 for i in range(len(node_ids_out_know))])

node_ids_know = list(node_ids_in_know)
node_ids_know.extend(node_ids_out_know)

node_data_know = node_data.loc[node_ids_know, :]

edgelist_know = edgelist[edgelist['target'].isin(
    node_ids_know) | edgelist['source'].isin(node_ids_know)]

G_know = sg.StellarGraph(nodes={"paper": node_data[feature_names]}, edges={
    "cites": edgelist_know})       # node_data_know[feature_names] not used since does not matter here
generator_noedge = FullBatchNodeGenerator(G_know, method="gcn")

target_encoding = get_target_encoding(
    node_data_know, node_label)       # To see later

test_gen_noedge = generator_noedge.flow(node_data_know.index, target_encoding)

pred = model.predict(test_gen_noedge)[0]

losses = np.sqrt(np.sum(np.abs(pred**2 - target_encoding**2), axis=1))

X = losses.reshape(-1, 1)
y = np.array(membership)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

clf = svm.SVR()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("#### Attack 3a Accuracy (losses) : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred > 0.5)))

X2 = pred
x_train, x_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.3)
clf = MLPClassifier(hidden_layer_sizes=(32, 16), random_state=1, max_iter=1000).fit(x_train, y_train)
y_pred = clf.predict(x_test)
clf = svm.SVR()
clf.fit(x_train, y_train)
y_pred2 = clf.predict(x_test)
print("#### Attack 3b Accuracy (preds) : {} / {}".format(100.0 * accuracy_score(y_test, y_pred > 0.5), 100.0 * accuracy_score(y_test, y_pred2 > 0.5)))

X3 = np.concatenate([X, X2], axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    X3, y, test_size=0.3)
clf = MLPClassifier(hidden_layer_sizes=(32, 16), random_state=1, max_iter=1000).fit(x_train, y_train)
y_pred = clf.predict(x_test)
clf = svm.SVR()
clf.fit(x_train, y_train)
y_pred2 = clf.predict(x_test)
print("#### Attack 3c Accuracy (preds) : {} / {}".format(100.0 * accuracy_score(y_test, y_pred > 0.5), 100.0 * accuracy_score(y_test, y_pred2 > 0.5)))

entropies = np.array(entr(pred).sum(
    axis=-1)/np.log(pred.shape[1]))
x_train, x_test, y_train, y_test = train_test_split(
    entropies.reshape(-1, 1), y, test_size=0.3)

clf = svm.SVR()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("#### Attack 3d Accuracy (Entropies) : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred > 0.5)))
