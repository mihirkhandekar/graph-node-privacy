from os import name
from model import GCNModel, get_target_encoding, get_train_data
from data import Data
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from plot import create_label_ratio_plot, plot_rocauc
from xgboost import XGBClassifier, XGBRegressor
from tensorflow.keras.callbacks import ModelCheckpoint
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from stellargraph import datasets, globalvar
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn import feature_extraction, model_selection, preprocessing, svm
from scipy.special import entr
import tensorflow as tf
import stellargraph as sg
import sklearn.metrics as skmet
import scipy.stats as stats
import pandas as pd
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


tf.get_logger().setLevel('WARNING')

cce = tf.keras.losses.CategoricalCrossentropy()


SEED = 10
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

DATASET_NAME = "pubmed"
SHADOW_DATASET_NAME = "cora"
method = 'gcn'

TRAIN = True
TRAIN_SHADOW = False
seq_epochs = {'citeseer': 25, 'cora': 35, 'facebook':35}

def load_and_train_model(dataset_name, test_acc_graph, degrees, neighbours):
    # Getting Graph Data
    dataset = Data(dataset_name)
    epochs, in_ratio, dropout, layer_sizes, activations = dataset.get_params()
    node_data, node_label, edgelist, feature_names = dataset.get_data()

    if dataset_name not in degrees:
        degrees[dataset_name] = {}
        neighbours[dataset_name] = {}
        for node in node_data.index:
            e = edgelist[(edgelist['target'] == node) | (edgelist['source'] == node)]
            degrees[dataset_name][node] = str(len(e.index))
            neighbours[dataset_name][node] = [str(ee) for ee in e.index]

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
            verbose=1,
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
        test_acc_graph[dataset_name] = str(test_metrics[1])
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
        verbose=1,
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
    test_acc_seq[model_name] = str(test_metrics[1])
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
    model.add(Dense(4, input_dim=x_in.shape[1], activation='tanh'))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(y_in.shape[1], activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(x_in, y_in, epochs=seq_epochs[model_name], batch_size=32,
              validation_data=[x_out, y_out])

    train_metrics = model.evaluate(x_in, y_in)
    print("\nIN Set Metrics:", model.metrics_names, train_metrics)
    test_metrics = model.evaluate(x_out, y_out)
    print("\nOUT Set Metrics:", model.metrics_names, test_metrics)
    test_acc_seq[model_name] = str(test_metrics[1])

    graph_attack(model, x_in, x_out, y_in, y_out, node_data_in.index,
                 node_data_out.index, acc_dict, model_name)#, loss='crossentropy')


def graph_attack(model, train_gen, test_gen, train_encoding, target_encoding, in_index, out_index, acc_dict, model_name, graph=None, loss='rms'):
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
        #squared = np.square(in_ - train_encoding)
        #rms_loss_train = np.sqrt(np.sum(squared, axis=1) * (1/np.array(in_).shape[1]))
        rms_loss_train = np.sqrt(np.sum(np.abs(in_**2 - train_encoding**2), axis=1))

        #squared = np.square(out_ - target_encoding)
        #rms_loss_test = np.sqrt(np.sum(squared, axis=1) * (1/np.array(out_).shape[1]))
        rms_loss_test = np.sqrt(np.sum(np.abs(out_**2 - target_encoding**2), axis=1))
    else:
        for label, pred in zip(train_encoding, in_):
            pred /= pred.sum(axis=-1, keepdims=True)
            rms_loss_train.append(
                np.sum(label * -np.log(pred), axis=-1, keepdims=False))

        for label, pred in zip(target_encoding, out_):
            pred /= pred.sum(axis=-1, keepdims=True)
            rms_loss_test.append(
                np.sum(label * -np.log(pred), axis=-1, keepdims=False))

    print('IN', np.mean(rms_loss_train), np.std(rms_loss_train), 'OUT', np.mean(rms_loss_test), np.std(rms_loss_test))
    min_len = int(min(len(rms_loss_train), len(rms_loss_test)) * 0.8)

    '''    x_train = np.concatenate(
            [rms_loss_train[:min_len], rms_loss_test[:min_len]])
        y_train = np.concatenate(
            [[1. for _ in range(min_len)], [0. for _ in range(min_len)]])
        x_test = np.concatenate(
            [rms_loss_train[min_len:], rms_loss_test[min_len:]])
        y_test = np.concatenate(
            [[1. for _ in range(len(rms_loss_train[min_len:]))], [0. for _ in range(len(rms_loss_test[min_len:]))]])
        indices = np.concatenate([in_index[min_len:], out_index[min_len:]])
    '''
    x = np.concatenate(
            [rms_loss_train[:min_len], rms_loss_test[:min_len]])
    y = np.concatenate(
            [[1. for _ in range(min_len)], [0. for _ in range(min_len)]])
    ind = np.concatenate([in_index[:min_len], out_index[:min_len]])
    x_train, x_test, y_train, y_test, _, indices = train_test_split(
        x, y, ind, test_size=0.25)


    clf = svm.SVR()
    clf.fit(x_train.reshape(-1, 1), y_train)
    y_pred = clf.predict(x_test.reshape(-1, 1))
    acc_dict[model_name] = str(roc_auc_score(y_test, y_pred > 0.5))

    if graph:
        index_real_mem, index_pred_mem, index_loss = graph
        if model_name not in index_real_mem:
            index_real_mem[model_name] = {}
            index_pred_mem[model_name] = {}
            index_loss[model_name] = {}
        for loss, membership, pred_membership, index in zip(x_test, y_test, y_pred, indices):
            index = str(index)
            if index in index_real_mem[model_name]:
                index_loss[model_name][index].append(str(loss))
                index_pred_mem[model_name][index].append(str(pred_membership))
                index_real_mem[model_name][index].append(str(membership))
            else:
                index_loss[model_name][index] = [str(loss)]
                index_pred_mem[model_name][index] = [str(pred_membership)]
                index_real_mem[model_name][index] = [str(membership)]

    return clf.support_vectors_[0][0]


#models = ['facebook', 'cora', 'citeseer', 'pubmed', 'financial']
models = ['cora', 'citeseer', 'facebook']
#models = ['citeseer']
attack_auc_seq = {}
test_acc_seq = {}
attack_auc_graph = {}
test_acc_graph = {}
attack_auc_seqs = []
test_acc_seqs = []
attack_auc_graphs = []
test_acc_graphs = []
degrees = {}
neighbours = {}

index_real_mem = {}
index_pred_mem = {}
index_loss = {}

loss_thresholds = {}

num_rounds = 20

for _ in range(num_rounds):
    for model_name in models:
        model, train_gen, val_gen, node_ids_in, node_ids_out, node_data, edgelist, node_data_in, node_data_out, node_label, feature_names, generator = load_and_train_model(
            model_name, test_acc_graph, degrees, neighbours)

        train_encoding = get_target_encoding(node_data_in, node_label)
        train_gen = generator.flow(node_data_in.index, train_encoding)
        target_encoding = get_target_encoding(node_data_out, node_label)
        test_gen = generator.flow(node_data_out.index, target_encoding)
        threshold = graph_attack(model, train_gen, test_gen, train_encoding, target_encoding,
                    node_data_in.index, node_data_out.index, attack_auc_graph, model_name, graph=(index_real_mem, index_pred_mem, index_loss))
            
        if model_name not in loss_thresholds:
            loss_thresholds[model_name] = [threshold]
        else:
            loss_thresholds[model_name].append(threshold)

        train_seq_seq_model(node_data_in, node_data_out, node_label, feature_names,
                            model_name, edgelist, node_data, attack_auc_seq, test_acc_seq)

        print('Sequential Attack AUC', attack_auc_seq)
        attack_auc_seqs.append(attack_auc_seq)
        print('Sequential Model accuracy', test_acc_seq)
        test_acc_seqs.append(test_acc_seq)
        print('Graph Attack AUC', attack_auc_graph)
        attack_auc_graphs.append(attack_auc_graph)
        print('Graph Model accuracy', test_acc_graph)
        test_acc_graphs.append(test_acc_graph)

        with open('records/dataset/{}_index_real_mem.json'.format(model_name), 'w') as f, open('records/dataset/{}_index_pred_mem.json'.format(model_name), 'w') as f2, open('records/dataset/{}_index_loss.json'.format(model_name), 'w') as f3:
            json.dump(index_real_mem[model_name], f)
            json.dump(index_pred_mem[model_name], f2)
            json.dump(index_loss[model_name], f3)

        with open('records/dataset/{}_degrees.json'.format(model_name), 'w') as f, open('records/dataset/{}_neighbours.json'.format(model_name), 'w') as f2, open('records/dataset/{}_thresholds.json'.format(model_name), 'w') as f3:
            json.dump(degrees[model_name], f)
            json.dump(neighbours[model_name], f2)
            json.dump(loss_thresholds[model_name], f3)

        with open('records/attack_auc_seq.json', 'w') as f_attack_auc_seq, open('records/test_acc_seq.json', 'w') as f_test_acc_seq, open('records/attack_auc_graph.json', 'w') as f_attack_auc_graph, open('records/test_acc_graph.json', 'w') as f_test_acc_graph:
            json.dump(attack_auc_seqs, f_attack_auc_seq)
            json.dump(test_acc_seqs, f_test_acc_seq)
            json.dump(attack_auc_graphs, f_attack_auc_graph)
            json.dump(test_acc_graphs, f_test_acc_graph)


