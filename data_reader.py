import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_data(filepath, my_test_size=0.2):
    print("Getting data from {0}".format(filepath))
    file_content = pd.read_csv(filepath) # odzywam sie po naglowkach
    file_content.replace(
                to_replace=["yes", "no"], 
                value=["spam", "not_spam"], 
                inplace=True) 
    x_train, x_test, y_train, y_test = train_test_split(
    file_content.drop(labels=["target"], axis = 1), 
    file_content["target"], 
    test_size=my_test_size, 
    random_state=42)
    y_test.replace('yes', 'spam')
    return x_train, x_test, y_train, y_test