from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
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