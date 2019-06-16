from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt     


def display(confusion, labels, name=""):
    print("Displaying!")
    cm = confusion
    print(cm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title("Confusion matrix of the {0} classifier".format(name))
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()