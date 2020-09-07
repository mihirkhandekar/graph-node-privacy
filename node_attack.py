#!/usr/bin/env python
# coding: utf-8

# # Stellargraph example: Graph Convolutional Network (GCN) on the CORA citation dataset

# Import NetworkX and stellar:

# In[1]:


import networkx as nx
import pandas as pd
import os

import stellargraph as sg
import numpy as np
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN

from stellargraph import globalvar

import itertools
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def load_pubmed(data_dir):
    edgelist = pd.read_csv(
        os.path.join(data_dir, "Pubmed-Diabetes.DIRECTED.cites.tab"),
        sep="\t",
        skiprows=2,
        header=None,
    )
    edgelist.drop(columns=[0, 2], inplace=True)
    edgelist.columns = ["source", "target"]
    # delete unneccessary prefix
    edgelist["source"] = edgelist["source"].map(lambda x: x.lstrip("paper:"))
    edgelist["target"] = edgelist["target"].map(lambda x: x.lstrip("paper:"))
    edgelist["label"] = "cites"  # set the edge type

    # Load the graph from the edgelist
    g_nx = nx.from_pandas_edgelist(edgelist, edge_attr="label")

    # Load the features and subject for each node in the graph
    nodes_as_dict = []
    with open(
        os.path.join(os.path.expanduser(data_dir), "Pubmed-Diabetes.NODE.paper.tab")
    ) as fp:
        for line in itertools.islice(fp, 2, None):
            line_res = line.split("\t")
            pid = line_res[0]
            feat_name = ["pid"] + [l.split("=")[0] for l in line_res[1:]][
                :-1
            ]  # delete summary
            feat_value = [l.split("=")[1] for l in line_res[1:]][:-1]  # delete summary
            feat_value = [pid] + [
                float(x) for x in feat_value
            ]  # change to numeric from str
            row = dict(zip(feat_name, feat_value))
            nodes_as_dict.append(row)

    # Create a Pandas dataframe holding the node data
    node_data = pd.DataFrame(nodes_as_dict)
    node_data.fillna(0, inplace=True)
    node_data["label"] = node_data["label"].astype(int)
    node_data["label"] = node_data["label"].astype(str)

    node_data.index = node_data["pid"]
    node_data.drop(columns=["pid"], inplace=True)
    node_data.head()

    for nid in node_data.index:
        g_nx.nodes[nid][globalvar.TYPE_ATTR_NAME] = "paper"  # specify node type

    feature_names = list(node_data.columns)
    feature_names.remove("label")

    return g_nx, node_data, feature_names, edgelist


# ### Loading the CORA network

# In[3]:


d = 'cora'

if d == 'cora':
    dataset = datasets.Cora()
    display(HTML(dataset.description))
    dataset.download()
    edgelist = pd.read_csv(
        os.path.join(dataset.data_directory, "cora.cites"),
        sep="\t",
        header=None,
        names=["target", "source"],
    )
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names = feature_names + ["subject"]
    node_data = pd.read_csv(
        os.path.join(dataset.data_directory, "cora.content"),
        sep="\t",
        header=None,
        names=column_names,
    )
    node_label = "subject"
elif d == 'pubmed':
    example_dataset = datasets.PubMedDiabetes()
    example_dataset.download()
    Gnx, node_data, feature_names, edgelist = load_pubmed(example_dataset.data_directory)
    edgelist = edgelist.drop(columns=['label'])
    node_label = "label"


# In[4]:


set(node_data[node_label])


# ### Splitting the data

# For machine learning we want to take a subset of the nodes for training, and use the rest for validation and testing. We'll use scikit-learn again to do this.
# 
# Here we're taking 140 node labels for training, 500 for validation, and the rest for testing.

# In[5]:


train_data, test_data = model_selection.train_test_split(
    node_data, train_size=500, test_size=None, stratify=node_data[node_label]
)
val_data, test_data = model_selection.train_test_split(
    test_data, train_size=64, test_size=None, stratify=test_data[node_label]
)
print(len(train_data), len(val_data), len(test_data),)


# In[6]:


print(train_data)


# Note using stratified sampling gives the following counts:

# In[7]:


from collections import Counter

Counter(train_data[node_label])


# The training set has class imbalance that might need to be compensated, e.g., via using a weighted cross-entropy loss in model training, with class weights inversely proportional to class support. However, we will ignore the class imbalance in this example, for simplicity.

# ### Converting to numeric arrays

# For our categorical target, we will use one-hot vectors that will be fed into a soft-max Keras layer during training. To do this conversion ...

# In[8]:


target_encoding = feature_extraction.DictVectorizer(sparse=False)

train_targets = target_encoding.fit_transform(train_data[[node_label]].to_dict("records"))
val_targets = target_encoding.transform(val_data[[node_label]].to_dict("records"))
test_targets = target_encoding.transform(test_data[[node_label]].to_dict("records"))
node_targets = target_encoding.transform(node_data[[node_label]].to_dict("records"))
node_features = node_data[feature_names]


# In[9]:


print(len(node_features), len(node_targets))
target_dict = {}
for ind, tar in enumerate(node_features.iterrows()):
    target_dict[tar[0]] = node_targets[ind]


# We now do the same for the node attributes we want to use to predict the subject. These are the feature vectors that the Keras model will use as input. The CORA dataset contains attributes 'w_x' that correspond to words found in that publication. If a word occurs more than once in a publication the relevant attribute will be set to one, otherwise it will be zero.

# In[10]:


train_nodes = train_data.index.values.tolist()
edgelist_noedge = edgelist[0:0]
edgelist_rest = edgelist[0:0]

for i, edge in edgelist.iterrows():
    if edge['target'] in train_nodes or edge['source'] in train_nodes:
        edgelist_noedge = edgelist_noedge.append(edgelist.iloc[i])
    else:
        edgelist_rest = edgelist_rest.append(edgelist.iloc[i])


# In[11]:


'''remove_n = int(0.4 * len(edgelist_noedge))
drop_indices = np.random.choice(edgelist_noedge.index, remove_n, replace=False)

l = edgelist_noedge.index.isin(drop_indices)
deleted_edgelist = edgelist_noedge[l]
print(len(edgelist_noedge))
new_edgelist = edgelist_noedge.drop(drop_indices)
print(len(edgelist_noedge), len(deleted_edgelist))
print(new_edgelist)
deleted_edgelist = deleted_edgelist.append(edgelist_rest)
edgelist = new_edgelist'''
new_edgelist = edgelist_noedge
edgelist = new_edgelist


# ## Creating the GCN model in Keras

# Now create a StellarGraph object from the NetworkX graph and the node features and targets. It is StellarGraph objects that we use in this library to perform machine learning tasks on.

# In[12]:


#print(new_edgelist)#print(np.sort(train_nodes))


# In[13]:


G = sg.StellarGraph(nodes={"paper": node_features}, edges={"cites": new_edgelist})


# In[14]:


print(G)


# In[15]:


print(G.info())


# To feed data from the graph to the Keras model we need a generator. Since GCN is a full-batch model, we use the `FullBatchNodeGenerator` class to feed node features and the normalized graph Laplacian matrix to the model.
# 
# Specifying the `method='gcn'` argument to the `FullBatchNodeGenerator` will pre-process the adjacency matrix and supply the normalized graph Laplacian matrix to the model.

# In[16]:


generator = FullBatchNodeGenerator(G, method="gcn")


# For training we map only the training nodes returned from our splitter and the target values.

# In[17]:


train_gen = generator.flow(train_data.index, train_targets)


# Now we can specify our machine learning model, we need a few more parameters for this:
# 
#  * the `layer_sizes` is a list of hidden feature sizes of each layer in the model. In this example we use two GCN layers with 16-dimensional hidden node features at each layer.
#  * `activations` is a list of activations applied to each layer's output
#  * `dropout=0.5` specifies a 50% dropout at each layer. 

# We create a GCN model as follows:

# In[18]:


gcn = GCN(
    layer_sizes=[128, 64,], activations=["tanh", "tanh", ], generator=generator,# dropout=0.2
)


# To create a Keras model we now expose the input and output tensors of the GCN model for node prediction, via the `GCN.build` method:

# In[19]:


x_inp, x_out = gcn.build()


# Finally we add a Keras `Dense` layer to use the node embeddings to predict the 7 categories.

# In[20]:


predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)


# ### Training the model

# Now let's create the actual Keras model with the input tensors `x_inp` and output tensors being the predictions `predictions` from the final dense layer

# In[21]:


model = Model(inputs=x_inp, outputs=predictions)
model.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)


# Train the model, keeping track of its loss and accuracy on the training set, and its generalisation performance on the validation set (we need to create another generator over the validation data for this)

# In[22]:


val_gen = generator.flow(val_data.index, val_targets)


# 
# Create callbacks for early stopping (if validation accuracy stops improving) and best model checkpoint saving:

# In[23]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

if not os.path.isdir("logs"):
    os.makedirs("logs")
es_callback = EarlyStopping(
    monitor="val_acc", patience=50
)  # patience is the number of epochs to wait before early stopping in case of no further improvement
mc_callback = ModelCheckpoint(
    "logs/best_model.h5" #, monitor="val_acc", save_best_only=True, save_weights_only=True
)


# Train the model

# In[24]:


history = model.fit_generator(
    train_gen,
    epochs=60,
    validation_data=val_gen,
    verbose=2,
    shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
    #callbacks=[es_callback, mc_callback],
    callbacks=[mc_callback]
)


# Plot the training history:

# In[25]:


sg.utils.plot_history(history)


# Reload the saved weights of the best model found during the training (according to validation accuracy)

# In[26]:


model.load_weights("logs/best_model.h5")


# Evaluate the best model on the test set

# In[27]:


test_gen = generator.flow(test_data.index, test_targets)


# In[28]:


test_metrics = model.evaluate_generator(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))


# In[29]:


###############################################
edgelist_noedge = edgelist[0:0]

G_noedge = sg.StellarGraph(nodes={"paper": node_features}, edges={"cites": edgelist_noedge})
generator_noedge = FullBatchNodeGenerator(G_noedge, method="gcn")

test_gen_noedge = generator_noedge.flow(test_data.index, test_targets)
train_gen_noedge = generator_noedge.flow(train_data.index, train_targets)


# In[30]:


t = model.predict_generator(test_gen_noedge)
#print(t)
#print(test_targets)
test_metrics = model.evaluate_generator(test_gen_noedge)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

t = t[0]
non_losses = np.sqrt(np.sum(np.abs(t**2 - test_targets**2), axis=1))
non_preds = t
#print('Losses', losses, np.mean(losses))

print(non_preds)


# In[31]:


t = model.predict_generator(train_gen_noedge)
#print(t[0][:10])

#print(train_targets[:10])
train_metrics = model.evaluate_generator(train_gen_noedge)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

t = t[0]
mem_losses = np.sqrt(np.sum(np.abs(t**2 - train_targets**2), axis=1))
mem_preds = t
#print('Losses', losses, np.mean(losses))
print(mem_preds)


# In[32]:


mem_mean = np.mean(mem_losses)
mem_var = np.var(mem_losses)

print('Members', mem_mean, mem_var)

non_mean = np.mean(non_losses)
non_var = np.var(non_losses)

print('Nontrain', non_mean, non_var)

import matplotlib.pyplot as plt
import scipy.stats as stats

import math

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


# In[33]:


###############################################################
import random
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


# In[34]:


print(len(mem_losses), len(non_losses), )


# In[35]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


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

from sklearn.model_selection import train_test_split
print('Splitting')
x_train, x_test, y_train, y_test, c_train, c_test = train_test_split(
    x3, y, c, test_size=0.7)

from sklearn import svm
clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("SVM Accuracy : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred)))

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

'''xmodel = XGBRegressor()
xmodel.fit(x_train, y_train)#, eval_set=[(x_test, y_test)], eval_metric='error')
y_pred = xmodel.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("XGB Accuracy: %.2f%%" % (accuracy * 100.0))'''


# In[36]:


print(np.shape(x), np.shape(x2), np.shape(x3))


# In[37]:


vals = y_pred == y_test
tr_d = {}
for v, ct in zip(vals, c_test):
    ct = ct[0]
    if ct not in tr_d:
        tr_d[ct] = [0, 0]
    
    if v:
        tr_d[ct][0] += 1
    
    tr_d[ct][1] += 1
    
print(tr_d)
    
tt = {}
for k, v in tr_d.items():
    if v[1] > 3:
        tt[k] = v[0]/v[1]
    
print(tt)


# In[38]:


tr_d = tt
barx = list(tr_d.keys())
bary = list(tr_d.values())
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = barx
students = bary
ax.bar(langs,students)
plt.show()


# In[39]:


###### PRECISION


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
    
print(tp_d)
    
tp = {}
for k, v in tp_d.items():
    
    if v[1] > 3:
        tp[k] = v[0]/(v[1])
    
print(tp)


# In[40]:


barx = list(tp.keys())
bary = list(tp.values())
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = barx
students = bary
ax.bar(langs,students)
plt.show()


# In[41]:


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
    
print(tn_d)
    
tn = {}
for k, v in tn_d.items():
    if v[1] > 3:
        tn[k] = v[0]/v[1]
    
print(tn)


# In[42]:


barx = list(tn.keys())
bary = list(tn.values())
import matplotlib.pyplot as plt
fig = plt.figure() 
ax = fig.add_axes([0,0,1,1])
langs = barx
students = bary
ax.bar(langs,students)
plt.show()


# In[43]:



def get_unused_sublist(node_id):
    sublist = deleted_edgelist[deleted_edgelist['target'] == node_id]
    sublist = sublist.append(deleted_edgelist[deleted_edgelist['source'] == node_id])
    return sublist


def get_target(node_id):
    '''for i, (index, data) in enumerate(node_features.iterrows()):
        if index == node_id:
            break
    return node_targets[i]'''
    '''print(node_id)
    tar = node_data.loc[int(node_id), node_label]
    ar = np.zeros((n_c))
    ar[int(tar) - 1] = 1.'''
    return target_dict[node_id]

def add_incorrect_edges(node_id, sublist, num):
    for _ in range(num):
        add_node_id = None
        while not add_node_id:
            add_node_id = nodes_list[np.random.randint(num_nodes)]
            if (((edgelist['target'] == node_id) & (edgelist['source'] == add_node_id)).any()) or (((edgelist['source'] == node_id) & (edgelist['target'] == add_node_id)).any()):
                add_node_id = None
                continue
        added_edge = {'source' : node_id, 'target' : add_node_id}
        sublist = sublist.append(added_edge, ignore_index=True)
        added_edge = {'target' : node_id, 'source' : add_node_id}
        sublist = sublist.append(added_edge, ignore_index=True)
    return sublist


# In[44]:


#################### Incremental Training -- ANALYSE CHANGE IN LOSS?
NUM_HOPS = 1
#NUM_INCORRECT_EDGES = 1
ATTACKER_NODE_KNOWLEDGE = 40
#PER_NODE_TRAIN_SAMPLES = 10

'''rand_nodes = random.sample(nodes_list, ATTACKER_NODE_KNOWLEDGE)
mem_losses = []
nmem_losses = []
non_losses = [] 
mem_preds = []
nmem_preds = []
non_preds = []
edgecounts = []
## Dataset creation
node_edge_counts = []
for nd in rand_nodes:
    sublist = get_k_edges(nd, k=NUM_HOPS)
    
    unused_sublist = get_unused_sublist(nd)
    row_target = get_target(nd)
    l = min(len(sublist), len(unused_sublist))
    if l == 0:
        rand_nodes.extend(random.sample(nodes_list, 1))
        continue
    else:
        edgecounts.append(l)

    known_sublist = sublist.iloc[:0][0:0]
    print('-----')
    print('KNOWN', (known_sublist))
    G_subset = sg.StellarGraph(nodes={"paper": node_features}, edges={"cites": known_sublist})
    generator_subset = FullBatchNodeGenerator(G_subset, method="gcn")
    test_gen_subset = generator_subset.flow([nd], [row_target])
    test_metrics_sub = model.evaluate_generator(test_gen_subset)
    known_loss = test_metrics_sub[0]
        
    for i in range(0, l):
        # Member added
        subsublist = sublist.iloc[i:i+1]
        print('KNOWN+MEM', (subsublist))
        G_subset = sg.StellarGraph(nodes={"paper": node_features}, edges={"cites": subsublist})
        generator_subset = FullBatchNodeGenerator(G_subset, method="gcn")
        test_gen_subset = generator_subset.flow([nd], [row_target])
        test_metrics_sub = model.evaluate_generator(test_gen_subset)
        print('Mem : ', model.predict_generator(test_gen_subset), row_target, nd)
        mem_preds.append(model.predict_generator(test_gen_subset))
        mem_losses.append(test_metrics_sub[0] - known_loss)
        node_edge_counts.append(len(get_k_edges(nd, k=0)))
                
        # Nonmember added
        subsublist = known_sublist.append(unused_sublist.iloc[i:i+1])
        print('KNOWN+NMEM', (subsublist))
        G_subset = sg.StellarGraph(nodes={"paper": node_features}, edges={"cites": subsublist})
        generator_subset = FullBatchNodeGenerator(G_subset, method="gcn")
        test_gen_subset = generator_subset.flow([nd], [row_target])
        test_metrics_sub = model.evaluate_generator(test_gen_subset)
        non_preds.append(model.predict_generator(test_gen_subset))
        print('NMem : ', model.predict_generator(test_gen_subset), row_target, nd)
        non_losses.append(test_metrics_sub[0] - known_loss)
        node_edge_counts.append(len(get_k_edges(nd, k=0)))'''


# nei = 3.
# 
# mem_losses_nei = [ml[1] for ml in mem_losses if ml[0] == nei]
# nmem_losses_nei = [ml[1] for ml in nmem_losses if ml[0] == nei]
# non_losses_nei = [ml[1] for ml in non_losses if ml[0] == nei]
# 
# x = [[mem_loss] for mem_loss in mem_losses_nei]
# x.extend([[nmem_loss] for nmem_loss in non_losses_nei])
# y = [1. for i in range(len(mem_losses_nei))]
# y.extend([0. for i in range(len(non_losses_nei))])
# 
# x, y = np.array(x), np.array(y)
# from sklearn.model_selection import train_test_split
# print('Splitting')
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.5)
# 
# from sklearn import svm
# clf = svm.SVC()
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# print("SVM Accuracy : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred)))
# 

# In[45]:


'''mem_preds = np.array(mem_preds)
nmem_preds = np.array(nmem_preds)
non_preds = np.array(non_preds)'''

print(edgecounts, np.sum(edgecounts))
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x = [mem_pred[0][0] for mem_pred in mem_preds]
x.extend([non_pred[0][0] for non_pred in non_preds])

x2 = [mem_loss for mem_loss in mem_losses]
x2.extend([non_loss for non_loss in non_losses])
x2 = np.expand_dims(x2, axis=1)

x3 = np.concatenate([x], axis=1)
'''
x3 = x
'''

y = [1. for i in range(len(mem_preds))]
y.extend([0. for i in range(len(non_preds))])
x = np.array(x3)
y = np.array(y)
x_train, x_test, y_train, y_test, c_train, c_test = train_test_split(
    x, y, node_edge_counts, test_size=0.2)
print('Splitting', len(y_train), len(y_test), len(x_train), len(x_test))

'''from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(8, 8, 4), max_iter=300, activation = 'relu', solver='adam',random_state=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Pred Accuracy : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred)))
'''
xmodel = XGBRegressor()
xmodel.fit(x_train, y_train)#, eval_set=[(x_test, y_test)], eval_metric='error')
y_pred = xmodel.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("XGB Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


print(c_test)
tru = (y_test == predictions)
print(tru)
less_count = 0
less_true = 0
more_count = 0
more_true = 0
for c, t in zip(c_test, tru):
    if c < 3:
        less_count = less_count + 1
        if t:
            less_true = less_true + 1
    else:
        more_count = more_count + 1
        if t:
            more_true = more_true + 1
            
print(less_true/less_count, more_true/more_count)


# In[ ]:


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==1 and y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==0 and y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return [TP, FP, TN, FN]

#print(y_test)
y_actual = [int(i) for i in y_test]
#print(predictions)
y_hat = [int(i) for i in predictions]

print(y_actual, y_hat)
print(len(y_actual), len(y_hat))

perf = perf_measure(y_actual, y_hat)
print(perf)


# In[ ]:


print('MEMBERS', np.mean(mem_losses), len(mem_losses))
'''for loss in mem_losses:
    print(loss)'''
    
print('NON-MEMBERS', np.mean(nmem_losses), len(nmem_losses))
'''for loss in nmem_losses:
    print(loss)'''
print('NON-TRAIN', np.mean(non_losses), len(non_losses))


# In[ ]:


print(np.average(mem_losses), len(mem_losses))
print(np.average(nmem_losses), len(nmem_losses))
print(np.average(non_losses), len(non_losses))


# In[ ]:


mem_mean = np.mean(mem_losses)
mem_var = np.var(mem_losses)

print('Members', mem_mean, mem_var)

'''nmem_mean = np.mean(nmem_losses)
nmem_var = np.var(nmem_losses)

print('NMembers', nmem_mean, nmem_var)'''

non_mean = np.mean(non_losses)
non_var = np.var(non_losses)

print('Nontrain', non_mean, non_var)

import matplotlib.pyplot as plt
import scipy.stats as stats

import math

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


# In[ ]:


'''from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

x = [[mem_loss] for mem_loss in mem_losses]
x.extend([[nmem_loss] for nmem_loss in nmem_losses])
y = [1. for i in range(len(mem_losses))]
y.extend([0. for i in range(len(nmem_losses))])

x, y = np.array(x), np.array(y)
from sklearn.model_selection import train_test_split
print('Splitting')
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5)

from sklearn import svm
clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("SVM Accuracy : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred)))'''

'''xmodel = XGBClassifier()
xmodel.fit(x_train, y_train)
y_pred = xmodel.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("XGB Accuracy: %.2f%%" % (accuracy * 100.0))'''


# In[ ]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

x = [[mem_loss] for mem_loss in mem_losses]
x.extend([[non_loss] for non_loss in non_losses])

y = [1. for i in range(len(mem_losses))]
y.extend([0. for i in range(len(non_losses))])

x, y = np.array(x), np.array(y)
from sklearn.model_selection import train_test_split
print('Splitting')
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5)

from sklearn import svm
clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("SVM Accuracy : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred)))
'''
xmodel = XGBClassifier()
xmodel.fit(x_train, y_train)
y_pred = xmodel.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("XGB Accuracy: %.2f%%" % (accuracy * 100.0))

'''


# In[ ]:


print(len(mem_losses))
print(len(non_losses))


# In[ ]:


mem_predictions = np.squeeze(mem_predictions)
mem_targets = np.array(mem_targets)
nmem_predictions = np.squeeze(nmem_predictions)
nmem_targets = np.array(nmem_targets)
mem_lossarr = np.expand_dims(np.squeeze(mem_losses), axis=1)
nmem_lossarr = np.expand_dims(np.squeeze(nmem_losses), axis=1)

print(mem_predictions.shape, mem_targets.shape, nmem_predictions.shape, nmem_targets.shape, mem_lossarr.shape, nmem_lossarr.shape)

x_mem = np.concatenate([mem_predictions, mem_targets, mem_lossarr], axis=1)
x_nmem = np.concatenate([nmem_predictions, nmem_targets, nmem_lossarr], axis=1)
x = np.concatenate([x_mem, x_nmem])
y = [1. for i in range(len(mem_predictions))]
y.extend([0. for i in range(len(nmem_predictions))])
y = np.array(y)
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.8)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(16,7))
clf.fit(x_train, y_train)
print('Score', clf.score(x_test, y_test))


# In[ ]:


sa = [1, 2, 3]
sa[:5]
for i in range(1, 1):
    print('aaa')
