from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
import confusion_displayer as cd

def run(x, y, x_test, y_test, display=False):
    print("Bayes")
    bayes = MultinomialNB()

    bayes.fit(x,y)
    y_pred = bayes.predict(x_test)
    labels = y.unique()
    confusion = confusion_matrix(y_test, y_pred, labels)
    if display:
        cd.display(confusion, labels, 3, "Bayes")
    
    return confusion


def runRFE(x, y, x_test, y_test, display=False):
    print("Bayes with feature selection")
    bayes = MultinomialNB()
    selector = RFE(bayes, 5, step=1)
    selector.fit(x,y)
    y_pred = selector.predict(x_test)
    labels = y.unique()
    confusion = confusion_matrix(y_test, y_pred, labels)
    if display:
        cd.display(confusion, labels, 31, "Bayes with feature selection")
    
    return confusion