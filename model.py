import stellargraph as sg
from stellargraph.layer import GCN
from stellargraph.mapper import FullBatchNodeGenerator
from tensorflow.keras import Model, layers, losses, metrics, optimizers
from sklearn import feature_extraction, model_selection, preprocessing, svm
import os

TRAIN_SIZE = 0.8

if not os.path.isdir("logs"):
    os.makedirs("logs")

def get_model(node_features, edgelist, node_ids, node_targets, val_node_ids, val_targets):
    G = sg.StellarGraph(nodes={"paper": node_features},
                        edges={"cites": edgelist})

    generator = FullBatchNodeGenerator(G, method="gcn")

    train_gen = generator.flow(node_ids, node_targets)

    gcn = GCN(
        # dropout=0.2
        layer_sizes=[128, 64, ], activations=["tanh", "tanh", ], generator=generator,
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
    train_ratio = int(len(node_data.index) * TRAIN_SIZE)

    train_data, val_data = model_selection.train_test_split(
        node_data, train_size=train_ratio, test_size=None, stratify=node_data[node_label]
    )
    train_targets = get_target_encoding(train_data, node_label)
    val_targets = get_target_encoding(val_data, node_label)
    return train_data, val_data, train_targets, val_targets


def get_target_encoding(data, label_name):
    target_encoding = feature_extraction.DictVectorizer(sparse=False)
    return target_encoding.fit_transform(data[[label_name]].to_dict("records"))
