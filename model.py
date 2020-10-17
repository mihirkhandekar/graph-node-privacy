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
    def __init__(self, dropout=0.5, layer_sizes=[32, ], activations=['tanh'], lr=0.01) -> None:
        self.dropout = dropout
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.lr = lr

    def get_conv_model(self):
        if self.gcn:
            return self.gcn
        else:
            raise Exception('Model not built yet')

    def get_model(self, node_features, edgelist, node_ids, node_targets, val_node_ids, val_targets, method='gcn'):
        G = sg.StellarGraph(nodes={"paper": node_features},
                            edges={"cites": edgelist})

        generator = FullBatchNodeGenerator(G, method=method)

        train_gen = generator.flow(node_ids, node_targets)

        if method == 'gcn':
            self.gcn = GCN(             
                dropout=self.dropout,
                layer_sizes=self.layer_sizes, activations=self.activations, generator=generator,# kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
            )
        else:
            self.gcn = GAT(              # GAT option
                layer_sizes=self.layer_sizes, activations=self.activations, generator=generator,# kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
            )

        x_inp, x_out = self.gcn.build()
        predictions = layers.Dense(
            units=node_targets.shape[1], activation="softmax")(x_out)

        model = Model(inputs=x_inp, outputs=predictions)

        model.compile(
            optimizer=optimizers.Adam(lr=self.lr),
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
