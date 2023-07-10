import numpy as np
import pandas as pd
def train_test_split(df, train_percent,randomState):

    train = df.sample(frac=train_percent, random_state=randomState)
    x_train = train.drop(columns=0)
    y_train = train[0]
    test = df.drop(train.index)
    x_test = test.drop(columns=0)
    y_test = test[0]
    return x_train, x_test, y_train, y_test