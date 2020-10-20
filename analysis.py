import json
import os
import random
from collections import Counter
from functools import reduce
from os import name
from matplotlib import pyplot

import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.metrics as skmet
import stellargraph as sg
import tensorflow as tf
from scipy.special import entr
from sklearn import feature_extraction, model_selection, preprocessing, svm
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from stellargraph import datasets, globalvar
from stellargraph.layer import GCN
from stellargraph.mapper import FullBatchNodeGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from xgboost import XGBClassifier, XGBRegressor

from data import Data
from model import GCNModel, get_target_encoding, get_train_data
from plot import create_label_ratio_plot, plot_rocauc, plot_model_degree_histograms, create_network_graph

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# tf.get_logger().setLevel('WARNING')

cce = tf.keras.losses.CategoricalCrossentropy()


'''SEED = 10
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
'''
DATASET_NAME = "pubmed"
SHADOW_DATASET_NAME = "cora"
method = 'gcn'

TRAIN = True
TRAIN_SHADOW = False
seq_epochs = {'citeseer': 20, 'cora': 35, 'facebook': 20, 'pubmed': 200}


def load_and_train_model(dataset_name, test_acc_graph, degrees, neighbours):
    # Getting Graph Data
    dataset = Data(dataset_name)
    epochs, in_ratio, dropout, layer_sizes, activations = dataset.get_params()
    node_data, node_label, edgelist, feature_names = dataset.get_data()
    # in_ratio = 0.5
    if dataset_name not in degrees:
        degrees[dataset_name] = {}
        neighbours[dataset_name] = {}
        for node in node_data.index:
            e = edgelist[(edgelist['target'] == node) |
                         (edgelist['source'] == node)]
            degrees[dataset_name][node] = len(e.index)
            neighbours[dataset_name][node] = list(e.index)

    num_nodes = len(node_data.index)
    num_edges = len(edgelist.index)

    # Splitting nodes into IN and OUT
    node_ids = list(node_data.index)  # List of all index of nodes
    num_nodes_in = int(in_ratio * num_nodes)
    node_ids_in = list(random.sample(node_ids, num_nodes_in))
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
    gcnmodel = GCNModel(dropout=dropout, layer_sizes=layer_sizes,
                        activations=activations)
    model, train_gen, val_gen, generator = gcnmodel.get_model(
        node_data[feature_names], edgelist_in, train_data.index, train_targets, val_data.index, val_targets, method=method)
    # Although all Node IDs are used,
    # only the IN edgelist is used for training

    if TRAIN:
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            # verbose=0,
            shuffle=False,
            callbacks=[ModelCheckpoint(
                "logs/{}_best_model.h5".format(dataset_name))]
        )
        target_encoding = get_target_encoding(node_data_out, node_label)
        test_gen = generator.flow(node_data_out.index, target_encoding)

        test_metrics = model.evaluate_generator(test_gen)
        print("\nOUT Set Metrics:")
        for name, val in zip(model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, val))
        test_acc_graph[dataset_name] = test_metrics[1]
        train_metrics = model.evaluate_generator(train_gen)
        print("\nIN Set Metrics:")
        for name, val in zip(model.metrics_names, train_metrics):
            print("\t{}: {:0.4f}".format(name, val))
    else:
        model.load_weights("logs/{}_best_model.h5".format(dataset_name))

    return model, train_gen, val_gen, node_ids_in, node_ids_out, node_data, edgelist, node_data_in, node_data_out, node_label, feature_names, generator


def train_seq_gnn_model(node_data_in, node_data_out, node_label, feature_names, model_name, edgelist, node_data, acc_dict, test_acc_seq):
    dataset = Data(model_name)
    epochs, in_ratio, dropout, layer_sizes, activations = dataset.get_params()
    gcnmodel = GCNModel(dropout=dropout, layer_sizes=layer_sizes,
                        activations=activations)
    edgelist_none = edgelist[:]
    train_data, val_data, train_targets, val_targets = get_train_data(
        node_data_in, node_label)    # Split IN data into train and validation sets
    model, train_gen, val_gen, generator = gcnmodel.get_model(
        node_data[feature_names], edgelist_none, train_data.index, train_targets, val_data.index, val_targets, method=method)
    history = model.fit(
        train_gen,
        epochs=int(epochs),  # - 0.2 * epochs),
        validation_data=val_gen,
        # verbose=0,
        shuffle=False,
        callbacks=[ModelCheckpoint(
            "logs/{}_noedge_best_model.h5".format(model_name))]
    )

    G = sg.StellarGraph(nodes={"paper": node_data[feature_names]},
                        edges={"cites": edgelist})

    generator = FullBatchNodeGenerator(G, method=method)

    train_encoding = get_target_encoding(node_data_in, node_label)
    train_gen = generator.flow(node_data_in.index, train_encoding)

    target_encoding = get_target_encoding(node_data_out, node_label)
    test_gen = generator.flow(node_data_out.index, target_encoding)

    train_metrics = model.evaluate_generator(train_gen)
    print("\nIN Set Metrics:", model.metrics_names, train_metrics)
    test_metrics = model.evaluate_generator(test_gen)
    print("\nOUT Set Metrics:", model.metrics_names, test_metrics)
    test_acc_seq[model_name] = test_metrics[1]
    graph_attack(model, train_gen, test_gen, train_encoding, target_encoding,
                 node_data_in.index, node_data_out.index, acc_dict, model_name, loss='crossentropy')


def train_seq_seq_model(node_data_in, node_data_out, node_label, feature_names, model_name, edgelist, node_data, acc_dict, test_acc_seq):
    dataset = Data(model_name)
    epochs, in_ratio, dropout, layer_sizes, activations = dataset.get_params()

    x_in = node_data_in[feature_names].to_numpy()
    x_out = node_data_out[feature_names].to_numpy()

    y_in = get_target_encoding(node_data_in, node_label)
    y_out = get_target_encoding(node_data_out, node_label)

    model = Sequential()
    model.add(Dense(64, input_dim=x_in.shape[1], activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(y_in.shape[1], activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    epochs = seq_epochs[model_name]
    model.fit(x_in, y_in, epochs=epochs, batch_size=32,
              validation_data=[x_out, y_out])

    train_metrics = model.evaluate(x_in, y_in)
    print("\nIN Set Metrics:", model.metrics_names, train_metrics)
    test_metrics = model.evaluate(x_out, y_out)
    print("\nOUT Set Metrics:", model.metrics_names, test_metrics)
    test_acc_seq[model_name] = test_metrics[1]

    graph_attack(model, x_in, x_out, y_in, y_out, node_data_in.index,
                 node_data_out.index, acc_dict, model_name, loss='crossentropy')


def graph_attack(model, train_gen, test_gen, train_encoding, target_encoding, in_index, out_index, acc_dict, model_name, round=None, graph=None, loss='rms'):
    in_preds = model.predict(train_gen)
    out_preds = model.predict(test_gen)
    rms_loss_train = []
    rms_loss_test = []

    if len(in_preds.shape) == 3:
        in_ = in_preds[0]
        out_ = out_preds[0]
    else:
        in_ = in_preds
        out_ = out_preds

    if loss == 'rms':
        rms_loss_train = np.sqrt(
            np.sum(np.abs(in_**2 - train_encoding**2), axis=1))
        rms_loss_test = np.sqrt(
            np.sum(np.abs(out_**2 - target_encoding**2), axis=1))
    else:
        for label, pred in zip(train_encoding, in_):
            pred /= pred.sum(axis=-1, keepdims=True)
            rms_loss_train.append(
                np.sum(label * -np.log(pred), axis=-1, keepdims=False))

        for label, pred in zip(target_encoding, out_):
            pred /= pred.sum(axis=-1, keepdims=True)
            rms_loss_test.append(
                np.sum(label * -np.log(pred), axis=-1, keepdims=False))

    print('IN', np.mean(rms_loss_train), np.std(rms_loss_train),
          'OUT', np.mean(rms_loss_test), np.std(rms_loss_test))
    min_len = int(min(len(rms_loss_train), len(rms_loss_test)))

    x = np.concatenate(
        [rms_loss_train[:min_len], rms_loss_test[:min_len]])
    y = np.concatenate(
        [[1. for _ in range(min_len)], [0. for _ in range(min_len)]])
    ind = np.concatenate([in_index[:min_len], out_index[:min_len]])

    clf = svm.SVR()
    clf.fit(x.reshape(-1, 1), y)
    y_pred = clf.predict(x.reshape(-1, 1))
    acc_dict[model_name] = roc_auc_score(y, y_pred > 0.5)

    if graph:
        index_real_mem, index_pred_mem, index_loss, index_round = graph
        if model_name not in index_real_mem:
            index_real_mem[model_name] = {}
            index_pred_mem[model_name] = {}
            index_loss[model_name] = {}
            index_round[model_name] = {}
        for loss, membership, pred_membership, index in zip(x, y, y_pred, ind):
            if index in index_real_mem[model_name]:
                index_loss[model_name][index].append(loss)
                index_pred_mem[model_name][index].append(pred_membership)
                index_real_mem[model_name][index].append(membership)
                index_round[model_name][index].append(round)
            else:
                index_loss[model_name][index] = [loss]
                index_pred_mem[model_name][index] = [pred_membership]
                index_real_mem[model_name][index] = [membership]
                index_round[model_name][index] = [round]

    return clf.support_vectors_[0][0]


#models = ['facebook', 'cora', 'citeseer', 'pubmed', 'financial']
models = ['cora', 'citeseer', 'facebook']
#models = ['cora']
attack_auc_seqs = []
test_acc_seqs = []
attack_auc_graphs = []
test_acc_graphs = []
degrees = {}
neighbours = {}

index_real_mem = {}
index_pred_mem = {}
index_loss = {}
index_round = {}

loss_thresholds = {}

num_rounds = 1

for i in range(num_rounds):
    print('########################**************** ROUND ', i+1)
    attack_auc_seq = {}
    test_acc_seq = {}
    attack_auc_graph = {}
    test_acc_graph = {}

    for model_name in models:
        print('***************************** Model ', model_name)
        print('***************************** GRAPH ATTACK Model {} Round {}'.format(model_name, i+1))
        model, train_gen, val_gen, node_ids_in, node_ids_out, node_data, edgelist, node_data_in, node_data_out, node_label, feature_names, generator = load_and_train_model(
            model_name, test_acc_graph, degrees, neighbours)
        train_encoding = get_target_encoding(node_data_in, node_label)
        train_gen = generator.flow(node_data_in.index, train_encoding)
        target_encoding = get_target_encoding(node_data_out, node_label)
        test_gen = generator.flow(node_data_out.index, target_encoding)
        threshold = graph_attack(model, train_gen, test_gen, train_encoding, target_encoding,
                                 node_data_in.index, node_data_out.index, attack_auc_graph, model_name, round=i, graph=(index_real_mem, index_pred_mem, index_loss, index_round))

        if model_name not in loss_thresholds:
            loss_thresholds[model_name] = [threshold]
        else:
            loss_thresholds[model_name].append(threshold)

        print('***************************** SEQUENTIAL ATTACK Model {} Round {}'.format(model_name, i+1))
        train_seq_seq_model(node_data_in, node_data_out, node_label, feature_names,
                            model_name, edgelist, node_data, attack_auc_seq, test_acc_seq)

        print('Sequential Attack AUC     :', attack_auc_seq)
        print('Graph Attack AUC          :', attack_auc_graph)
        print('\nSequential Model accuracy :', test_acc_seq)
        print('Graph Model accuracy      :', test_acc_graph)

    attack_auc_seqs.append(attack_auc_seq)
    test_acc_seqs.append(test_acc_seq)
    attack_auc_graphs.append(attack_auc_graph)
    test_acc_graphs.append(test_acc_graph)


def convert_vals_to_str(dic):
    for k, v in dic.items():
        dic[k] = str(v)
    return dic


def add_to_confusion_dict(dic, real, pred):
    if real == 1.0:
        if pred < 0.5:  # False negative
            dic['fn'] += 1
        else:   # True positive
            dic['tp'] += 1
    elif real == 0.0:
        if pred < 0.5:  # true negative
            dic['tn'] += 1
        else:   # false positive
            dic['fp'] += 1


def find_max_subtree_len(node, round, checked_nodes, edges, node_labels):
    ns = neighbours[model_name][node]
    checked_nodes.append(node)
    lgth = 1

    if round in index_round[model_name][node]:
        pred = index_pred_mem[model_name][node][round]
        real = index_real_mem[model_name][node][round]
        if (real == 0.0 and pred > 0.5) or (real == 1.0 and pred <= 0.5):
            correct = 0
        else:
            correct = 1
        node_labels[node] = correct
        for nd in ns:
            if nd in index_pred_mem[model_name] and nd not in checked_nodes:
                neighbour_rounds = index_round[model_name][nd]
                if round in neighbour_rounds:  # If neighbour was present in ith round
                    edges.append([nd, node])
                    subtree_len = find_max_subtree_len(
                        nd, round, checked_nodes, edges, node_labels)
                    lgth += subtree_len
    return lgth


for model_name in models:
    print("####################### MODEL NAME #######################", model_name)
    # Plot prediction vs degree histogram (members vs non-members)
    plot_model_degree_histograms(
        degrees, index_pred_mem, index_real_mem, model_name)

    # See if neighbours of FP, FN are P or N
    # Are incorrectly classified nodes clustered together?
    fp = {'fn': 0, 'tn': 0, 'tp': 0, 'fp': 0}
    fn = {'fn': 0, 'tn': 0, 'tp': 0, 'fp': 0}

    for node, reals in index_real_mem[model_name].items():
        preds = index_pred_mem[model_name][node]
        ns = neighbours[model_name][node]
        for i, (real, pred) in enumerate(zip(reals, preds)):
            if real == 1.0 and pred < 0.5:
                # false negative case
                for neighbour in ns:
                    if neighbour in index_pred_mem[model_name]:
                        neighbour_rounds = index_round[model_name][neighbour]
                        if i in neighbour_rounds:  # If neighbour was present in ith round
                            round = neighbour_rounds.index(i)
                            nei_pred = index_pred_mem[model_name][neighbour][round]
                            nei_real = index_real_mem[model_name][neighbour][round]
                            add_to_confusion_dict(fn, nei_real, nei_pred)
            elif real == 0.0 and pred >= 0.5:
                # false positive case
                for neighbour in ns:
                    if neighbour in index_pred_mem[model_name]:
                        neighbour_rounds = index_round[model_name][neighbour]
                        if i in neighbour_rounds:   # If neighbour was present in ith round
                            round = neighbour_rounds.index(i)
                            nei_pred = index_pred_mem[model_name][neighbour][round]
                            nei_real = index_real_mem[model_name][neighbour][round]
                            add_to_confusion_dict(fp, nei_real, nei_pred)

    try:
        factor = 1.0/sum(fp.values())
        for k in fp:
            fp[k] = fp[k]*factor
        factor = 1.0/sum(fn.values())
        for k in fn:
            fn[k] = fn[k]*factor
    except:
        pass
    print('FALSE Positive nodes', fp)
    print('FALSE Negative nodes', fn)
    with open('analysis_records/{}_neighbour_dist_fp.json'.format(model_name), 'w') as f, open('analysis_records/{}_neighbour_dist_fn.json'.format(model_name), 'w') as f2:
        json.dump(fp, f)
        json.dump(fn, f2)

    # * Find largest subtree to draw
    max_depth = 0
    max_node, max_round = None, None
    max_edges, max_node_labels = None, None

    max_depth2 = 0
    max_node2, max_round2 = None, None
    max_edges2, max_node_labels2 = None, None

    max_subtrees = []

    for node, reals in index_real_mem[model_name].items():
        for round in range(num_rounds):
            edges = []
            node_labels = {}
            node_known_depth = find_max_subtree_len(
                node, round, [], edges, node_labels)
            if node_known_depth > max_depth:
                max_depth2 = max_depth
                max_node2 = max_node
                max_round2 = max_round
                if max_edges:
                    max_edges2 = list(max_edges)
                if max_node_labels:
                    max_node_labels2 = dict(max_node_labels)

                max_depth = node_known_depth
                max_node = node
                max_round = round
                max_edges = list(edges)
                max_node_labels = dict(node_labels)

    print(max_depth, max_node, max_round, max_edges, max_node_labels)
    print(max_depth2, max_node2, max_round2, max_edges2, max_node_labels2)

    create_network_graph(max_node_labels, max_edges, model_name, num=0)
    create_network_graph(max_node_labels2, max_edges2, model_name, num=1)


    # Is an incorrectly classified node incorrect in each iteration?
    accuracies = []
    for node, reals in index_real_mem[model_name].items():
        preds = index_pred_mem[model_name][node]
        ns = neighbours[model_name][node]
        acc = accuracy_score(reals, np.array(preds) > 0.5)
        accuracies.append(acc)
    bins = np.linspace(0, 1, 20)
    plt.hist(accuracies, bins, alpha=0.5, label='Node-wise accuracy')
    plt.legend(loc='upper right')
    plt.savefig('analysis_plots/' + model_name +
                '/node_acc', bbox_inches="tight")
    plt.clf()
    plt.cla()
