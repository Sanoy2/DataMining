import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import MLP
import DTC
import SVM
import Bayes
import KNN

import od_Pawla

import data_reader


def main():
    filepath = "spam.dat"
    x,X_test,y,Y_test = data_reader.get_data(filepath)

    confusion = MLP.runalt(x,y, X_test, Y_test, display=True, optimize=False)
    confusion = DTC.run(x,y, X_test, Y_test, display=True, optimize=False)
    # confusion = DTC.runRFE(x,y, X_test, Y_test, display=True)

    # confusion = SVM.run(x,y, X_test, Y_test, True)
    # confusion = Bayes.run(x,y, X_test, Y_test, True)
    confusion = KNN.runalt(x,y, X_test, Y_test, display=True, optimization=False)
    plt.show()


if __name__ == "__main__":
    main()