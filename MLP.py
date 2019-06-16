from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def run(x, y, x_test, y_test, display=False):
    print("Multilayer perceptron")
    mlp = MLPClassifier()

    mlp.fit(x,y)
    y_pred = mlp.predict(x_test)
    # print(y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    # print(confusion)
    return confusion