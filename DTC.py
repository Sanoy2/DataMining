from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


def run(x, y, x_test, y_test, display=False):
    print("Decision tree")
    dtc = DecisionTreeClassifier()

    dtc.fit(x,y)
    y_pred = dtc.predict(x_test)
    # print(y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    # print(confusion)
    return confusion



