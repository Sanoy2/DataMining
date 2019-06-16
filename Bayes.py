from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import confusion_displayer as cd

def run(x, y, x_test, y_test, display=False):
    print("Bayes")
    bayes = MultinomialNB()

    bayes.fit(x,y)
    y_pred = bayes.predict(x_test)
    labels = y.unique()
    confusion = confusion_matrix(y_test, y_pred, labels)
    # print(confusion)
    if display:
        cd.display(confusion, labels, "Bayes")
    
    return confusion
    