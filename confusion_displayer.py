from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt     


def display(confusion, labels, name=""):
    print("Displaying!")
    cm = confusion
    print(cm)

    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
    plt.show()