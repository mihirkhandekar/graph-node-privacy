import itertools
import os
import random
from platform import node

import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import datasets, globalvar

NUM_NODES = 10000
NUM_EDGES = 20000
FEATURE_SIZE = 100
NUM_CLASSES = 5


class Data:
    def __init__(self, d) -> None:
        self.d = d

    def load_er(self, num_nodes=NUM_NODES, num_edges=NUM_EDGES, feature_size=FEATURE_SIZE, num_classes=NUM_CLASSES):
        G = nx.gnm_random_graph(num_nodes, num_edges)
        nodes = G.nodes
        indices = []
        features = [[] for _ in range(feature_size)]
        labels = []
        for node in nodes:
            indices.append(node)
            for i in range(feature_size):
                features[i].append(np.random.randint(2))
            correct_label = np.random.randint(num_classes)
            labels.append(correct_label)

        feature_names = ["w_{}".format(ii) for ii in range(feature_size)]
        data = {}
        for i, name in enumerate(feature_names):
            data[name] = features[i]
        data['label'] = labels
        node_data = pd.DataFrame(
            data, columns=feature_names + ['label'], index=indices)
        node_label = 'label'
        edge1 = []
        edge2 = []
        for edge in G.edges:
            edge1.append(edge[0])
            edge2.append(edge[1])
        data = {'source': edge1, 'target': edge2}
        edgelist = pd.DataFrame(data, columns=['source', 'target'])
        return node_data, node_label, edgelist, feature_names

    def load_pubmed(self, data_dir):
        edgelist = pd.read_csv(
            os.path.join(data_dir, "Pubmed-Diabetes.DIRECTED.cites.tab"),
            sep="\t",
            skiprows=2,
            header=None,
        )
        edgelist.drop(columns=[0, 2], inplace=True)
        edgelist.columns = ["source", "target"]
        edgelist["source"] = edgelist["source"].map(
            lambda x: x.lstrip("paper:"))
        edgelist["target"] = edgelist["target"].map(
            lambda x: x.lstrip("paper:"))
        edgelist["label"] = "cites"  # set the edge type

        # Load the graph from the edgelist
        g_nx = nx.from_pandas_edgelist(edgelist, edge_attr="label")

        # Load the features and subject for each node in the graph
        nodes_as_dict = []
        with open(
            os.path.join(os.path.expanduser(data_dir),
                         "Pubmed-Diabetes.NODE.paper.tab")
        ) as fp:
            for line in itertools.islice(fp, 2, None):
                line_res = line.split("\t")
                pid = line_res[0]
                feat_name = ["pid"] + [l.split("=")[0] for l in line_res[1:]][
                    :-1
                ]  # delete summary
                feat_value = [l.split("=")[1]
                              for l in line_res[1:]][:-1]  # delete summary
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
            # specify node type
            g_nx.nodes[nid][globalvar.TYPE_ATTR_NAME] = "paper"

        feature_names = list(node_data.columns)
        feature_names.remove("label")

        return g_nx, node_data, feature_names, edgelist

    def get_data(self):
        '''
        # Data description
        ## node_data : Pandas dataframe of node features and labels.
        ## node_label : Name of label column in dataset.
        ## edgelist : Pandas dataframe of edge connections target-source
        ## feature_names : List of node feature names. Used to get node features from dataset.
        '''

        if self.d == 'cora':
            dataset = datasets.Cora()
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
                names=column_names
            )
            node_label = "subject"
        elif self.d == 'pubmed':
            dataset = datasets.PubMedDiabetes()
            dataset.download()
            Gnx, node_data, feature_names, edgelist = self.load_pubmed(
                dataset.data_directory)
            edgelist = edgelist.drop(columns=['label'])
            node_label = "label"
        elif self.d == 'financial':
            # Download to datasets/ folder using command `kaggle datasets download -d ellipticco/elliptic-data-set`
            node_data, classes, edgelist, feature_names = self.get_financial_data()
            node_data = node_data.join(classes)
            node_label = 'class'
        elif self.d == 'er':
            # Download to datasets/ folder using command `kaggle datasets download -d ellipticco/elliptic-data-set`
            node_data, node_label, edgelist, feature_names = self.load_er()
        elif self.d == 'citeseer':
            dataset = datasets.CiteSeer()
            dataset.download()
            edgelist = pd.read_csv(
                os.path.join(dataset.data_directory, "citeseer.cites"),
                sep="\t",
                header=None,
                names=["target", "source"],
            )
            edgelist.target.apply(str)
            edgelist.source.apply(str)
            feature_names = ["w_{}".format(ii) for ii in range(3703)]
            column_names = feature_names + ["subject"]
            node_data = pd.read_csv(
                os.path.join(dataset.data_directory, "citeseer.content"),
                sep="\t",
                header=None,
                names=column_names
            )
            node_data.index = node_data.index.map(str)
            node_label = "subject"
            all_sources = edgelist['source'].tolist()
            all_targets = edgelist['target'].tolist()
            edge_nodes = list(set(all_sources + all_targets))
            feature_nodes = list(node_data.index)

            diff = list(np.setdiff1d(edge_nodes, feature_nodes))
            edgelist_del = edgelist[(edgelist['target'].isin(
                diff)) | (edgelist['source'].isin(diff))]

            edgelist = pd.concat([edgelist, edgelist_del]
                                 ).drop_duplicates(keep=False)
        elif self.d == 'facebook':
            node_data, edgelist, feature_names, node_label = self.get_facebook_data()

        else:
            raise Exception('No such dataset', self.d)

        return node_data, node_label, edgelist, feature_names

    def get_facebook_data(self):
        # https://github.com/benedekrozemberczki/datasets
        edgelist = pd.read_csv(
            os.path.join("datasets/facebook",
                         "facebook_edges.csv"),
            sep=",",
            header=0,
            names=["target", "source"],
        )
        feature_names = ["w_{}".format(ii) for ii in range(4714)]
        node_data = pd.read_csv(
            os.path.join("datasets/facebook",
                         "features.csv"),
            sep=",",
            index_col=0,
            header=0,
            names=feature_names,
            low_memory=False
        )
        node_data.index = node_data.index.astype('int32')
        classes = pd.read_csv(
            os.path.join("datasets/facebook",
                         "facebook_target.csv"),
            sep=",",
            index_col=0,
            header=0,
            # usecols=['page_type']
        )
        classes.index = classes.index.astype('int32')
        classes.drop(['facebook_id', 'page_name'], axis=1)
        label_name = 'page_type'
        node_data = node_data.join(classes)
        return node_data, edgelist, feature_names, label_name

    def get_financial_data(self):
        edgelist = pd.read_csv(
            os.path.join("datasets/elliptic_bitcoin_dataset",
                         "elliptic_txs_edgelist.csv"),
            sep=",",
            header=0,
            names=["target", "source"],
        )
        feature_names = ["w_{}".format(ii) for ii in range(166)]
        node_data = pd.read_csv(
            os.path.join("datasets/elliptic_bitcoin_dataset",
                         "elliptic_txs_features.csv"),
            sep=",",
            index_col=0,
            header=None,
            names=feature_names,
        )

        classes = pd.read_csv(
            os.path.join("datasets/elliptic_bitcoin_dataset",
                         "elliptic_txs_classes.csv"),
            sep=",",
            index_col=0,
            header=0,
        )
        return node_data, classes, edgelist, feature_names

    def get_params(self):
        if self.d == 'facebook':
            epochs = 50
            in_ratio = 0.02
            dropout = 0.
            layer_sizes = [64, ]
            activations = ['tanh', ]
        elif self.d == 'cora':
            epochs = 35
            in_ratio = 0.2
            dropout = 0.
            layer_sizes = [16, ]
            activations = ['tanh']
        elif self.d == 'citeseer':
            epochs = 50
            in_ratio = 0.2
            dropout = 0
            layer_sizes = [32, ]
            activations = ['tanh', ]
        elif self.d == 'pubmed':
            epochs = 250
            in_ratio = 0.03
            dropout = 0.
            layer_sizes = [32, ]
            activations = ['tanh', ]
        elif self.d == 'financial':
            epochs = 500
            in_ratio = 0.01
            dropout = 0.
            layer_sizes = [32, ]
            activations = ['tanh', ]

        return epochs, in_ratio, dropout, layer_sizes, activations


if __name__ == "__main__":
    ds = Data('er')
    node_data, node_label, edgelist, feature_names = ds.get_data()
    print('Nodes', len(node_data.index), "Edges", len(edgelist.index))
    print(node_data)
    print(edgelist)
