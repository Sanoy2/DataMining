import pandas as pd
import numpy as np


import MLP
import LR
import DTC
import SVM
import Bayes

import data_reader


def main():
    filepath = "spam.dat"
    x,X_test,y,Y_test = data_reader.get_data(filepath)
    
    # confusion = MLP.run(x,y, X_test, Y_test)
    # confusion = LR.run(x,y, X_test, Y_test) # nie dziala
    # confusion = DTC.run(x,y, X_test, Y_test)
    # confusion = SVM.run(x,y, X_test, Y_test)
    confusion = Bayes.run(x,y, X_test, Y_test, True)


if __name__ == "__main__":
    main()