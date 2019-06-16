import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import MLP
import LR
import DTC
import SVM
import Bayes
import KNN

import od_Pawla

import data_reader


def main():
    filepath = "spam.dat"
    x,X_test,y,Y_test = data_reader.get_data(filepath)

    # confusion = MLP.run(x,y, X_test, Y_test, display=True, optimize=False)
    # confusion = LR.run(x,y, X_test, Y_test, True) # nie dziala
    # confusion = DTC.run(x,y, X_test, Y_test, True)
    # confusion = SVM.run(x,y, X_test, Y_test, True)
    # confusion = Bayes.run(x,y, X_test, Y_test, True)
    # confusion = KNN.run(x,y, X_test, Y_test, display=True)
    # hidden_layers = mpl_optimizer.run(x,y, X_test, Y_test)
    # confusion = MLP.run(x,y, X_test, Y_test, True, hidden_layers)
    # od_Pawla.SVM(x,y, X_test, Y_test)
    plt.show()



if __name__ == "__main__":
    main()