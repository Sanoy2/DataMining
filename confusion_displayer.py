from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import judge

def display(confusion, labels, number, name=""):
    print("Displaying!")
    cm = confusion
    print(cm)

    plt.figure(number)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    result = judge.get_points(confusion)
    result = -result * 100
    result = round(result, 4)
    if result > 100:
        result = "Just really bad.."
    ax.set_title("Confusion matrix of the {0} classifier, \nresult: {1} % of bad classification".format(name, result))
    ax.xaxis.set_ticklabels(labels) 
    ax.yaxis.set_ticklabels(labels)