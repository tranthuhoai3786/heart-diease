import numpy as np
import json


class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000) -> None:
        self.lr = learning_rate
        self.n_iter = iterations

    def fit(self, X: np.array, Y: np.array):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.n_iter):
            self.update_weights()
        return self

    def update_weights(self):
        sigmoid = 1 / (1 + np.exp(-(np.dot(self.X, self.w) + self.b)))
        tmp = sigmoid - self.Y.T
        tmp = np.reshape(tmp, self.m)
        dW = np.dot(self.X.T, tmp) / self.m
        db = np.sum(tmp) / self.m

        self.w -= self.lr * dW
        self.b -= self.lr * db
        return self

    def predict(self, X: np.array) -> np.array:
        Z = 1 / (1 + np.exp(-(np.dot(X, self.w) + self.b)))
        Y = np.where(Z > 0.5, 1, 0)
        return Y

    def save(self, f_path: str) -> None:
        d = self.__dict__
        del d["X"], d["Y"]
        d = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()}
        json_obj = json.dumps(d, ensure_ascii=False, indent=4)
        with open(f_path, "w") as f:
            f.write(json_obj)

    @staticmethod
    def load(f_path: str):
        result = None
        with open(f_path, "r") as f:
            result = json.loads(f.read())
        result.w = np.array(result.w)
        return result


if __name__ == "__main__":
    from ConfusionMatrix import *
    from correlation import *
    from sqlalchemy import create_engine
    import pandas as pd
    import math

    def train_test_split(X, y, splitting_factor):
        n_train = math.floor(splitting_factor * X.shape[0])
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train:]
        y_test = y[n_train:]
        return X_train, X_test, y_train, y_test

    con_str = "mysql+mysqldb://root:@localhost/heart_disease_health_indicators"
    con = create_engine(con_str, echo=False)

    df = pd.read_sql("SELECT * FROM data", con=con)

    X = df.drop(columns=["HeartDiseaseorAttack"]).values
    y = df["HeartDiseaseorAttack"].values
    print("Shape of data:", X.shape)
    print("Shape of label:", y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, 0.8)
    print("Shape of train data:", X_train.shape)
    print("Shape of test data:", X_test.shape)

    clf = MyLogisticRegression()
    clf.fit(X_train, Y_train)
    clf.save("model_logistic.json")
    predictions = clf.predict(X_test)
    result = pd.DataFrame({"Actual": Y_test, "Predict": predictions})

    tp_rf, tn_rf, fp_rf, fn_rf = compute_tp_tn_fn_fp(Y_test, predictions)
    acc = compute_accuracy(tp_rf, tn_rf, fn_rf, fp_rf)
    precision_ = compute_precision(tp_rf, fp_rf)
    recall = compute_recall(tp_rf, fn_rf)
    f1_score = compute_f1_score(Y_test, predictions)
    print("Accuracy for Logistic Regression:", acc)
    print("Precision for Logistic Regression:", precision_)
    print("Recall for Logistic Regression:", recall)
    print("F1 score for Logistic Regression:", f1_score)

    arr = {
        "id": [1],
        "acc": [acc],
        "precision_": [precision_],
        "recall": [recall],
        "f1_score": [f1_score],
    }
    arr = pd.DataFrame(arr)
    arr.to_sql(name="evaluatescratchlr", con=con, if_exists="replace", index=False)
    x = np.array([1, 2, 3])
    x.T
