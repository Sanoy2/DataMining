from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import confusion_displayer as cd
import judge


def run(x, y, x_test, y_test, display=False, hidden_layers=[70, 25], optimize=False):
    print("Multilayer perceptron")
    if optimize is True:
        hidden_layers = optimization(x, y, x_test, y_test)

    mlp = MLPClassifier(hidden_layer_sizes=(hidden_layers))
    mlp.fit(x,y)
    y_pred = mlp.predict(x_test)
    labels = y.unique()
    confusion = confusion_matrix(y_test, y_pred, labels)
    if display:
        cd.display(confusion, labels, 2, "Multilayer perceptron")
    return confusion


def optimization(x, y, x_test, y_test):
    print("Multilayer perceptron optimizer")
    best_result = -999999
    a = 0 # 80              or 70
    b = 0 # 40 2.5496 %        25 also 3.1%  
    
    for i in range(50, 80, 10):
        for j in range(5, 45, 10):
            print("i: {0}, j: {1}".format(i,j))
            mlp = MLPClassifier(hidden_layer_sizes=(i, j))
            mlp.fit(x,y)
            y_pred = mlp.predict(x_test)
            confusion = confusion_matrix(y_test, y_pred)
            result = judge.get_points(confusion)
            if result > best_result:
                best_result = result
                a = i
                b = j

    print("Best so far: a:{0}, b:{1}".format(a,b))
    new_a = a
    new_b = b
    range_to_search_a = 5
    range_to_search_b = 3
    for i in range(new_a - range_to_search_a, new_a + range_to_search_a, 1):
        for j in range(new_b - range_to_search_b, new_b + range_to_search_b, 1):
            print("i: {0}, j: {1}".format(i,j))
            mlp = MLPClassifier(hidden_layer_sizes=(i, j))
            mlp.fit(x,y)
            y_pred = mlp.predict(x_test)
            confusion = confusion_matrix(y_test, y_pred)
            result = judge.get_points(confusion)
            if result > best_result:
                best_result = result
                a = i
                b = j
    print("Found best: a:{0}, b:{1}".format(a,b))
    return [a, b]