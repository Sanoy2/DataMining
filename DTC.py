from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn import tree
import numpy as np
import confusion_displayer as cd
import judge

def runRFE(x, y, x_test, y_test, display=False):
    print("Decision tree with feature selection")
    dtc = get_best_so_far()
    selector = RFE(dtc, 5, step=1)
    selector.fit(x,y)

    y_pred = selector.predict(x_test)
    labels = y.unique()
    confusion = confusion_matrix(y_test, y_pred, labels)
    if display:
        cd.display(confusion, labels, 90, "Decision tree with feature selection")

    return confusion


def run(x, y, x_test, y_test, display=False, optimize=False):
    print("Decision tree")
    
    if optimize is True:
        dtc = optimization(x, y, x_test, y_test)
    else:
        dtc = get_best_so_far()
        dtc.fit(x,y)

    y_pred = dtc.predict(x_test)
    labels = y.unique()
    confusion = confusion_matrix(y_test, y_pred, labels)
    if display:
        cd.display(confusion, labels, 4, "Decision tree")

    return confusion


def optimization(x, y, x_test, y_test):
    rmse_scorer = make_scorer(error_scorer, greater_is_better=True)
    
    clf_tree=tree.DecisionTreeClassifier()
    parameters = {
        'criterion':['gini','entropy'],
        'max_depth': [1,3,5,7,9,11,13,15],
        'min_samples_split' : range(10,500,20),
        'min_samples_leaf' : range(1,50)
    }
    gs = GridSearchCV(DecisionTreeClassifier(), param_grid=parameters, scoring=rmse_scorer, n_jobs=3, cv=5)
    gs = gs.fit(x, y)
    print(-gs.best_score_)
    print(gs.best_params_)
    my_model = gs.best_estimator_
    return my_model


def error_scorer(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    return judge.get_points(confusion)


def get_best_so_far():
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=30, min_samples_leaf=4)
    # dtc = DecisionTreeClassifier()
    return dtc