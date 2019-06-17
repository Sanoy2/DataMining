from sklearn import svm as _svm
from sklearn.metrics import confusion_matrix
import confusion_displayer as cd


def run(x, y, x_test, y_test, display=False, g="auto"):
    print("SVM")
    svm = _svm.SVC(gamma=g) # g = auto / scale
    svm.fit(x,y)
    y_pred = svm.predict(x_test)
    labels = y.unique()
    confusion = confusion_matrix(y_test, y_pred, labels)
    if display:
        cd.display(confusion, labels, 6, "SVM")
    return confusion