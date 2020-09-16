import itertools
import os

import networkx as nx
import pandas as pd
from stellargraph import datasets, globalvar


class Data:
    def __init__(self, d) -> None:
        self.d = d

    def load_pubmed(self, data_dir):
        edgelist = pd.read_csv(
            os.path.join(data_dir, "Pubmed-Diabetes.DIRECTED.cites.tab"),
            sep="\t",
            skiprows=2,
            header=None,
        )
        edgelist.drop(columns=[0, 2], inplace=True)
        edgelist.columns = ["source", "target"]
        edgelist["source"] = edgelist["source"].map(lambda x: x.lstrip("paper:"))
        edgelist["target"] = edgelist["target"].map(lambda x: x.lstrip("paper:"))
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
                names=column_names,
            )
            node_label = "subject"
        elif self.d == 'pubmed':
            dataset = datasets.PubMedDiabetes()
            dataset.download()
            Gnx, node_data, feature_names, edgelist = self.load_pubmed(
                dataset.data_directory)
            edgelist = edgelist.drop(columns=['label'])
            node_label = "label"
        elif self.d == 'blogcatalog3':
            dataset = datasets.BlogCatalog3()
            dataset.download()
            print(dataset.data_directory)

        return node_data, node_label, edgelist, feature_names

if __name__ == "__main__":
    ds = Data('blogcatalog3')
    data = ds.get_data()
