import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sklearn.metrics as skmet
from networkx.drawing.nx_agraph import graphviz_layout, pygraphviz_layout


def plot_rocauc(y_test, y_pred, name):
    fpr, tpr, threshold = skmet.roc_curve(y_test, y_pred)
    roc_auc = skmet.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'r', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('plots/' + name)
    plt.clf()
    plt.cla()


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_model_degree_histograms(degrees, index_pred_mem, index_real_mem, model_name):
    degree_pred_mem = {}
    degree_pred_nonmem = {}
    print(len(index_pred_mem[model_name]), len(degrees[model_name]))
    for node, preds in index_pred_mem[model_name].items():
        degree = degrees[model_name][node]
        membership = index_real_mem[model_name][node]
        for m, l in zip(membership, preds):
            if m == 1.:
                if degree in degree_pred_mem:
                    degree_pred_mem[degree].append(l)
                else:
                    degree_pred_mem[degree] = [l]
            else:
                if degree in degree_pred_nonmem:
                    degree_pred_nonmem[degree].append(l)
                else:
                    degree_pred_nonmem[degree] = [l]

    plt_degrees = []
    plt_mempred = []
    plt_nmempred = []

    for degree, m_preds in degree_pred_mem.items():
        if degree in degree_pred_nonmem:
            nm_preds = degree_pred_nonmem[degree]
            bins = np.linspace(0, 1, 20)
            plt.hist(m_preds, bins, alpha=0.5, label='Members')
            plt.hist(nm_preds, bins, alpha=0.5, label='Non-Members')
            plt.legend(loc='upper right')
            plt.savefig('analysis_plots/' + model_name +
                        '/degree' + str(degree), bbox_inches="tight")
            plt.clf()
            plt.cla()


def create_label_ratio_plot(plt_labels, plt_scores, plt_ratio, name):
    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(plt_labels))
    rects1 = ax.bar(x - width/2, plt_scores, width, label='AUC')
    rects2 = ax.bar(x + width/2, plt_ratio, width, label='Label Ratio')
    ax.set_ylabel('Scores vs Label ratio in dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(plt_labels)
    ax.legend()
    autolabel(rects1, ax)
    autolabel(rects2, ax)

    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment='right')

    # fig.tight_layout()
    plt.savefig('plots/' + name, bbox_inches="tight")
    plt.clf()
    plt.cla()


def create_network_graph(max_node_labels, max_edges, model_name, num=0):
    sg = nx.Graph()

    correct_nodelist = []
    incorrect_nodelist = []
    plt_labels = {}
    for node_id, label in max_node_labels.items():
        plt_labels[node_id] = node_id
        if label == 1:
            correct_nodelist.append(node_id)
        else:
            incorrect_nodelist.append(node_id)

    sg.add_nodes_from(correct_nodelist)
    sg.add_nodes_from(incorrect_nodelist)

    pos = nx.spring_layout(sg)

    nx.draw_networkx_nodes(sg, pos, nodelist=correct_nodelist, node_color="b")
    nx.draw_networkx_nodes(
        sg, pos, nodelist=incorrect_nodelist, node_color="r")
    nx.draw_networkx_edges(sg, pos, edgelist=max_edges, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(sg, pos, plt_labels, font_size=10)

    plt.savefig('analysis_plots/' + model_name +
                '/max_subgraph' + str(num), bbox_inches="tight")
    plt.clf()
    plt.cla()
