import numpy as np


class SupportVectorMachinesClassifier:
    # khoi tao learning rate, Lambda parameter and the number of iterations
    def __init__(self, learning_rate=0.001, lambda_param=0.01, iterations=1000) -> None:
        self.lr = learning_rate
        self.n_iter = iterations
        self.lp = lambda_param

    def fit(self, X: np.array, Y: np.array):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        for i in range(self.n_iter):
            self.update_weights()

    def update_weights(self):
        y_label = np.where(self.Y <= 0, -1, 1)
        for i, x_i in enumerate(self.X):
            condition = y_label[i] * (np.dot(x_i,self.w) - self.b) >= 1
            if condition:
                dw = 2 * self.lp * self.w
                db = 0
            else:
                dw = 2 * self.lp * self.w - np.dot(x_i, y_label[i])
                db = y_label[i]
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict(self, X: np.array) -> np.array:
        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        Y = np.where(predicted_labels <= -1, 0, 1)
        return Y
