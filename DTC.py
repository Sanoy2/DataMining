from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import confusion_displayer as cd


def run(x, y, x_test, y_test, display=False):
    print("Decision tree")
    dtc = DecisionTreeClassifier()

    dtc.fit(x,y)
    y_pred = dtc.predict(x_test)
    labels = y.unique()
    confusion = confusion_matrix(y_test, y_pred, labels)
    if display:
        cd.display(confusion, labels, 4, "Decision tree")

    return confusion



