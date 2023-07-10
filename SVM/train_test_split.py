import math
def train_test_split(X, y, splitting_factor):
    n_train = math.floor(splitting_factor * X.shape[0])
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    return X_train, X_test, y_train, y_test