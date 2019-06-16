from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def run(x, y, x_test, y_test, display=False):
    print("Logistic regression")
    lr = LogisticRegression()

    lr.fit(x, y)
    y_pred = lr.predict(y_test)
    confusion = confusion_matrix(y_test, y_pred)
    # print(confusion)
    return confusion
