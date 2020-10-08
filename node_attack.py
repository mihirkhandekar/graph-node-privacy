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

dataset = Data('cora')
node_data, node_label, edgelist, feature_names = dataset.get_data()

train_data, test_data = model_selection.train_test_split(
    node_data, train_size=500, test_size=None, stratify=node_data[node_label]
)
val_data, test_data = model_selection.train_test_split(
    test_data, train_size=64, test_size=None, stratify=test_data[node_label]
)

target_encoding = feature_extraction.DictVectorizer(sparse=False)

train_targets = target_encoding.fit_transform(
    train_data[[node_label]].to_dict("records"))
val_targets = target_encoding.transform(
    val_data[[node_label]].to_dict("records"))
test_targets = target_encoding.transform(
    test_data[[node_label]].to_dict("records"))
node_targets = target_encoding.transform(
    node_data[[node_label]].to_dict("records"))
node_features = node_data[feature_names]


print(len(node_features), len(node_targets))
target_dict = {}
for ind, tar in enumerate(node_features.iterrows()):
    target_dict[tar[0]] = node_targets[ind]


train_nodes = train_data.index.values.tolist()
edgelist_noedge = edgelist[0:0]
edgelist_rest = edgelist[0:0]

for i, edge in edgelist.iterrows():
    if edge['target'] in train_nodes or edge['source'] in train_nodes:
        edgelist_noedge = edgelist_noedge.append(edgelist.iloc[i])
    else:
        edgelist_rest = edgelist_rest.append(edgelist.iloc[i])


new_edgelist = edgelist_noedge
edgelist = new_edgelist


G = sg.StellarGraph(nodes={"paper": node_features},
                    edges={"cites": new_edgelist})

generator = FullBatchNodeGenerator(G, method="gcn")

train_gen = generator.flow(train_data.index, train_targets)

gcn = GCN(
    # dropout=0.2
    layer_sizes=[128, 64, ], activations=["tanh", "tanh", ], generator=generator,
)

x_inp, x_out = gcn.build()

predictions = layers.Dense(
    units=train_targets.shape[1], activation="softmax")(x_out)

model = Model(inputs=x_inp, outputs=predictions)
model.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)

val_gen = generator.flow(val_data.index, val_targets)

if not os.path.isdir("logs"):
    os.makedirs("logs")
es_callback = EarlyStopping(
    monitor="val_acc", patience=50
)  
mc_callback = ModelCheckpoint(
    "logs/best_model.h5"  # , monitor="val_acc", save_best_only=True, save_weights_only=True
)

history = model.fit_generator(
    train_gen,
    epochs=50,
    validation_data=val_gen,
    verbose=2,
    shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
    #callbacks=[es_callback, mc_callback],
    callbacks=[mc_callback]
)

sg.utils.plot_history(history)

model.load_weights("logs/best_model.h5")

test_gen = generator.flow(test_data.index, test_targets)


test_metrics = model.evaluate_generator(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))


###############################################
edgelist_noedge = edgelist[0:0]

G_noedge = sg.StellarGraph(nodes={"paper": node_features}, edges={
                           "cites": edgelist_noedge})
generator_noedge = FullBatchNodeGenerator(G_noedge, method="gcn")

test_gen_noedge = generator_noedge.flow(test_data.index, test_targets)
train_gen_noedge = generator_noedge.flow(train_data.index, train_targets)


t = model.predict_generator(test_gen_noedge)

test_metrics = model.evaluate_generator(test_gen_noedge)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

t = t[0]
non_losses = np.sqrt(np.sum(np.abs(t**2 - test_targets**2), axis=1))
non_preds = t

print(non_preds)


t = model.predict_generator(train_gen_noedge)

train_metrics = model.evaluate_generator(train_gen_noedge)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

t = t[0]
mem_losses = np.sqrt(np.sum(np.abs(t**2 - train_targets**2), axis=1))
mem_preds = t
#print('Losses', losses, np.mean(losses))
print(mem_preds)


mem_mean = np.mean(mem_losses)
mem_var = np.var(mem_losses)

print('Members', mem_mean, mem_var)

non_mean = np.mean(non_losses)
non_var = np.var(non_losses)

print('Nontrain', non_mean, non_var)


mem_sigma = np.sqrt(mem_var)
#nmem_sigma = np.sqrt(nmem_var)
non_sigma = np.sqrt(non_var)

plt.plot(sorted(mem_losses), stats.norm.pdf(
    sorted(mem_losses), mem_mean, mem_sigma))
'''plt.plot(sorted(nmem_losses), stats.norm.pdf(
    sorted(nmem_losses), nmem_mean, nmem_sigma))
'''
plt.plot(sorted(non_losses), stats.norm.pdf(
    sorted(non_losses), non_mean, non_sigma))
plt.savefig('plots/losses')
plt.clf()
plt.cla()


###############################################################
nodes_list = sorted(list(node_data.index))
num_nodes = len(nodes_list)
n_c = len(set(node_data[node_label]))


def get_edges(node_id):
    neighbours = []
    sublist = edgelist[edgelist['target'] == node_id]
    sublist = sublist.append(edgelist[edgelist['source'] == node_id])
    neighbours.extend(list(sublist['target']))
    neighbours.extend(list(sublist['source']))
    neighbours = list(set(neighbours) - set([node_id]))
    return sublist, neighbours


def get_k_edges(node_id, k):
    sublist, neighbours = get_edges(node_id)
    if k > 0:
        for neighbour in neighbours:
            sublist = sublist.append(get_k_edges(neighbour, k-1))
    return sublist.drop_duplicates()


train_nodes = train_data.index.values.tolist()
test_nodes = test_data.index.values.tolist()
train_nodes_deg = []
for node in train_nodes:
    train_nodes_deg.append(len(get_k_edges(node, 0)))

test_nodes_deg = []
for node in test_nodes:
    test_nodes_deg.append(len(get_k_edges(node, 0)))


print(len(mem_losses), len(non_losses), )


x = [[mem_loss] for mem_loss in mem_losses]
x.extend([[non_loss] for non_loss in non_losses[:len(mem_losses)]])

x2 = [mem_pred for mem_pred in mem_preds]
x2.extend([non_pred for non_pred in non_preds[:len(mem_losses)]])

print('Shapes', np.shape(x), np.shape(x2))

x3 = np.concatenate([x], axis=1)


y = [1. for i in range(len(mem_losses))]
y.extend([0. for i in range(len(mem_losses))])

c = [[l] for l in train_nodes_deg]
c.extend([[l] for l in test_nodes_deg[:len(mem_losses)]])


x3, y = np.array(x3), np.array(y)

print('Splitting 0.3')
x_train, x_test, y_train, y_test, c_train, c_test = train_test_split(
    x3, y, c, test_size=0.7)

'''clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("SVM Accuracy : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred)))
'''
clf = svm.SVR()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
#print(y_test, y_pred)
print("SVM Accuracy : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred > 0.5)))


fpr, tpr, threshold = skmet.roc_curve(y_test, y_pred)
roc_auc = skmet.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('plots/rocauc')
plt.clf()
plt.cla()


'''xmodel = XGBRegressor()
# , eval_set=[(x_test, y_test)], eval_metric='error')
xmodel.fit(x_train, y_train)
y_pred = xmodel.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("XGB Accuracy: %.2f%%" % (accuracy * 100.0))
'''



y_test_dict = {}
y_pred_dict = {}
for c, real, pred in zip(np.array(c_test).flatten(), y_test, y_pred):
    if c in y_test_dict:
        y_test_dict[c].append(real)
    else:
        y_test_dict[c] = [real]
    
    if c in y_pred_dict:
        y_pred_dict[c].append(pred)
    else:
        y_pred_dict[c] = [pred]
    
for key in y_test_dict:
    #print('Key', key, ":", y_test_dict[key], y_pred_dict[key])
    fpr, tpr, threshold = skmet.roc_curve(y_test_dict[key], y_pred_dict[key])
    roc_auc = skmet.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.savefig('plots/rocauc_' + str(key))
    plt.clf()
    plt.cla()

y_test_dict = {}
y_pred_dict = {}
y_test_dict[0] = []
y_test_dict[1] = []
y_pred_dict[0] = []
y_pred_dict[1] = []

for c, real, pred in zip(np.array(c_test).flatten(), y_test, y_pred):
    if c <= 3:
        y_test_dict[0].append(real)
        y_pred_dict[0].append(pred)
    else:
        y_test_dict[1].append(real)
        y_pred_dict[1].append(pred)

print(y_test_dict)
print(y_pred_dict)

fpr, tpr, threshold = skmet.roc_curve(y_test_dict[0], y_pred_dict[0])
roc_auc = skmet.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('plots/rocauc_small')
plt.clf()
plt.cla()


print(y_test_dict[1], y_pred_dict[1])
fpr, tpr, threshold = skmet.roc_curve(y_test_dict[1], y_pred_dict[1])
roc_auc = skmet.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'r', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('plots/rocauc_big')
plt.clf()
plt.cla()

vals = y_pred == y_test
tr_d = {}
for v, ct in zip(vals, c_test):
    ct = ct[0]
    if ct not in tr_d:
        tr_d[ct] = [0, 0]

    if v:
        tr_d[ct][0] += 1

    tr_d[ct][1] += 1

print('=========tr_d')
for key in sorted(tr_d.keys()):
    print("%s: %s" % (key, tr_d[key]))


tt = {}
for k, v in tr_d.items():
    if v[1] > 3:
        tt[k] = v[0]/v[1]

print('=========tt')
for key in sorted(tt.keys()):
    print("%s: %s" % (key, tt[key]))


tr_d = tt
barx = list(tr_d.keys())
bary = list(tr_d.values())
fig = plt.figure()
ax = fig.add_subplot(111)
#ax = fig.add_axes([0, 0, 1, 1])
langs = barx
students = bary
ax.bar(langs, students)
plt.savefig('plots/fig1')
plt.clf()
plt.cla()


# PRECISION


vals = y_pred == y_test
tp_d = {}
for v, ct, pr in zip(vals, c_test, y_pred):
    ct = ct[0]
    if ct not in tp_d:
        tp_d[ct] = [0, 0]

    if v and pr:
        tp_d[ct][0] += 1

    if pr:
        tp_d[ct][1] += 1

print('==========tp_d')
for key in sorted(tp_d.keys()):
    print("%s: %s" % (key, tp_d[key]))

tp = {}
for k, v in tp_d.items():
    if v[1] > 3:
        tp[k] = v[0]/(v[1])

print('============tp')
for key in sorted(tp.keys()):
    print("%s: %s" % (key, tp[key]))


barx = list(tp.keys())
bary = list(tp.values())
fig = plt.figure()
ax = fig.add_subplot(111)
#ax = fig.add_axes([0, 0, 1, 1])
langs = barx
students = bary
ax.bar(langs, students)
plt.savefig('plots/fig2')
plt.clf()
plt.cla()


vals = y_pred == y_test
tn_d = {}
for v, ct, pr in zip(vals, c_test, y_pred):
    ct = ct[0]
    if ct not in tn_d:
        tn_d[ct] = [0, 0]

    if v and not pr:
        tn_d[ct][0] += 1

    if not pr:
        tn_d[ct][1] += 1

print('============tn_d')
for key in sorted(tn_d.keys()):
    print("%s: %s" % (key, tn_d[key]))

tn = {}
for k, v in tn_d.items():
    if v[1] > 3:
        tn[k] = v[0]/v[1]

print('=============tn')
for key in sorted(tn.keys()):
    print("%s: %s" % (key, tn[key]))


barx = list(tn.keys())
bary = list(tn.values())
fig = plt.figure()
ax = fig.add_subplot(111)
#ax = fig.add_axes([0, 0, 1, 1])
langs = barx
students = bary
ax.bar(langs, students)
plt.savefig('plots/fig3')
