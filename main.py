import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import MLP

filepath = "spam.dat"

def main():
    print("Hello")
    print(filepath)
    file_content = pd.read_csv(filepath) # odzywam sie po naglowkach
    X_train, X_test, y_train, y_test = train_test_split(
    file_content.drop(labels=["target"], axis = 1), 
    file_content["target"], 
    test_size=0.2, 
    random_state=42)

    MLP.mlp(X_train,y_train, X_test, y_test)


if __name__ == "__main__":
    main()