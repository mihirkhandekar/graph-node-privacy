import matplotlib.pyplot as plt
import sklearn.metrics as skmet
import numpy as np

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

    #fig.tight_layout()
    plt.savefig('plots/' + name, bbox_inches="tight")
    plt.clf()
    plt.cla()
