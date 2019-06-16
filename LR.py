from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import confusion_displayer as cd


def run(x, y, x_test, y_test, display=False):
    print("Logistic regression")
    lr = LogisticRegression()
    lr.fit(a, b)
    y_pred = lr.predict(d)
    labels = b.unique()
    confusion = confusion_matrix(d, y_pred, labels)
    if display:
        cd.display(confusion, labels, 1, "Logistic regression")

    return confusion
