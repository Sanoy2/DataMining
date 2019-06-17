from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np
import confusion_displayer as cd
import judge


def run(x, y, x_test, y_test, display=False, neighbors = 3, optimization=False):
    print("K-NN classifier")
    if optimization is True:
        neighbors = optimize(x, y, x_test, y_test)

    knn = KNeighborsClassifier(neighbors)
    knn.fit(x,y)
    y_pred = knn.predict(x_test)
    labels = y.unique()
    confusion = confusion_matrix(y_test, y_pred, labels)
    if display:
        cd.display(confusion, labels, 9, "{0}-NN classifier".format(neighbors))
    return confusion


def optimize(x, y, x_test, y_test):
    best_result = -99999999
    best_number = 1
    for i in range(1, 20, 1):
        print(i)
        knn = KNeighborsClassifier(i)
        knn.fit(x,y)
        y_pred = knn.predict(x_test)
        confusion = confusion_matrix(y_test, y_pred)
        result = judge.get_points(confusion)
        if result > best_result:
            best_result = result
            best_number = i
    print("Best neighbors number = {0}".format(best_number))
    return best_number


def runalt(x, y, x_test, y_test, display=False, optimization=False):
    print("K-NN classifier")
    if optimization is True:
        knn = optimizationalt(x, y, x_test, y_test)
    else:
        knn = get_best_so_far()
        knn.fit(x,y)
    
    y_pred = knn.predict(x_test)
    labels = y.unique()
    confusion = confusion_matrix(y_test, y_pred, labels)
    if display:
        cd.display(confusion, labels, 9, "k-NN classifier")
    return confusion


def optimizationalt(x, y, x_test, y_test):
    rmse_scorer = make_scorer(error_scorer, greater_is_better=True)
    parameter_space = {
        'n_neighbors' : range(3,10),
        'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'weights' : ['uniform', 'distance'],
        'leaf_size' : range(10,50,5)
    }
    gs = GridSearchCV(KNeighborsClassifier(), param_grid=parameter_space, scoring=rmse_scorer, n_jobs = 3, cv=5)
    gs = gs.fit(x, y)
    print(-gs.best_score_)
    print(gs.best_params_)
    my_model = gs.best_estimator_
    return my_model


def get_best_so_far():
    knn = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance', leaf_size=10)
    # knn = KNeighborsClassifier()
    return knn

def error_scorer(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    return judge.get_points(confusion)