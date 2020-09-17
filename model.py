import os

import stellargraph as sg
from sklearn import feature_extraction, model_selection, preprocessing, svm
from stellargraph.layer import GCN, GAT
from stellargraph.mapper import FullBatchNodeGenerator
from tensorflow.keras import Model, layers, losses, optimizers, regularizers

TRAIN_RATIO = 0.9
target_encoding = feature_extraction.DictVectorizer(sparse=False)

if not os.path.isdir("logs"):
    os.makedirs("logs")


class GCNModel:
    def __init__(self, dropout=0.5, layer_sizes=[32, ], activations=['tanh']) -> None:
        self.dropout = dropout
        self.layer_sizes = layer_sizes
        self.activations = activations

    def get_model(self, node_features, edgelist, node_ids, node_targets, val_node_ids, val_targets):
        G = sg.StellarGraph(nodes={"paper": node_features},
                            edges={"cites": edgelist})

        generator = FullBatchNodeGenerator(G, method="gcn")

        train_gen = generator.flow(node_ids, node_targets)

        gcn = GAT(
            # dropout=self.dropout,
            layer_sizes=self.layer_sizes, activations=self.activations, generator=generator
        )

        x_inp, x_out = gcn.build()

        predictions = layers.Dense(
            units=node_targets.shape[1], activation="softmax")(x_out)

        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(
            optimizer=optimizers.Adam(lr=0.01),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )

        val_gen = generator.flow(val_node_ids, val_targets)

        return model, train_gen, val_gen, generator


def get_train_data(node_data, node_label):
    train_size = int(len(node_data.index) * TRAIN_RATIO)

    train_data, val_data = model_selection.train_test_split(
        node_data, train_size=train_size, test_size=None, stratify=node_data[node_label]
    )
    train_targets = get_target_encoding(train_data, node_label)
    val_targets = get_target_encoding(val_data, node_label)
    return train_data, val_data, train_targets, val_targets


def get_target_encoding(data, label_name):
    return target_encoding.fit_transform(data[[label_name]].to_dict("records"))
