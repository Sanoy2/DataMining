from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import confusion_displayer as cd
import judge

from sklearn.feature_selection import RFE
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


def runalt(x, y, x_test, y_test, display=False, optimize=False):
    print("Multilayer perceptron")
    if optimize is True:
        mlp = optimizationalt(x, y, x_test, y_test)
    else:
        mlp = get_best_so_far()
        mlp.fit(x,y)
    y_pred = mlp.predict(x_test)
    labels = y.unique()
    confusion = confusion_matrix(y_test, y_pred, labels)
    if display:
        cd.display(confusion, labels, 2, "Multilayer perceptron")
    return confusion


def optimizationalt(x, y, x_test, y_test):
    rmse_scorer = make_scorer(error_scorer, greater_is_better=True)
    parameter_space = {
        'hidden_layer_sizes': [(80,40), (70,25), (65,20,5)],
        # 'hidden_layer_sizes': [(80,40)],
        # 'max_iter' : [300],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        # 'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }
    gs = GridSearchCV(MLPClassifier(), param_grid=parameter_space, scoring=rmse_scorer, n_jobs=3, cv=5)
    gs = gs.fit(x, y)
    print(-gs.best_score_)
    print(gs.best_params_)
    my_model = gs.best_estimator_
    return my_model



def error_scorer(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    return judge.get_points(confusion)



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


def get_best_so_far():
    mlp = MLPClassifier(hidden_layer_sizes=(70,25), activation='relu', learning_rate='constant', max_iter=300, solver='adam')
    # mlp = MLPClassifier()
    return mlp