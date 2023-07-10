import pandas as pd
import numpy as np
import random

from ConfusionMatrix import *
from sqlalchemy import create_engine
from SVM import SupportVectorMachinesClassifier
from train_test_split import train_test_split

con_str = "mysql+mysqldb://root:123456@localhost/th3iot"
con = create_engine(con_str, echo=False)

df = pd.read_sql("SELECT * FROM test15k", con=con)
df =  df.head(10000)
X=df.iloc[:,1:].values
Y=df.iloc[:,0].values
# print("X = ", X)
# print("Y = ", Y)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X_train, X_test, y_train, y_test = train_test_split(X, Y, 0.8)
model = SupportVectorMachinesClassifier()
model.fit(X_train, y_train)
predict = model.predict(X_test)

# acc = accuracy(Y_test, Y_pred)
# print(acc)

tp_rf, tn_rf, fp_rf, fn_rf = compute_tp_tn_fn_fp(y_test, predict)
print('TP for SVM :', tp_rf)
print('TN for SVM :', tn_rf)
print('FP for SVM :', fp_rf)
print('FN for SVM :', fn_rf)

acc = compute_accuracy(tp_rf, tn_rf, fn_rf, fp_rf)
precision_ = compute_precision(tp_rf, fp_rf)
recall = compute_recall(tp_rf, fn_rf)
f1_score = compute_f1_score(y_test, predict)
print('Accuracy for SVM :', acc)
print('Precision for SVM :', precision_)
print('Recall for SVM :', recall)
print('F1 score for SVM :', f1_score)
arr = {
    "id": [1],
    "acc": [acc],
    "precision_": [precision_],
    "recall": [recall],
    "f1_score": [f1_score],
}
arr = pd.DataFrame(arr)
arr.to_sql(name="evaluatescratchsvm", con=con, if_exists="replace", index=False)