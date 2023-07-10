from ConfusionMatrix import *
from correlation import *
from sqlalchemy import create_engine
from LogisticRegression import MyLogisticRegression
import pandas as pd
import math


def train_test_split(X, y, splitting_factor):
    n_train = math.floor(splitting_factor * X.shape[0])
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    return X_train, X_test, y_train, y_test


con_str = "mysql+mysqldb://root:123456@localhost/th3iot"
con = create_engine(con_str, echo=False)

df = pd.read_sql("SELECT * FROM test15k", con=con)

X = df.drop(columns=["HeartDiseaseorAttack"]).values
y = df["HeartDiseaseorAttack"].values
print("Shape of data:", X.shape)
print("Shape of label:", y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, 0.8)
print("Shape of train data:", X_train.shape)
print("Shape of test data:", X_test.shape)

clf = MyLogisticRegression()
clf.fit(X_train, Y_train)
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