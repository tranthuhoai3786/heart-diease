import random
import pandas as pd
import numpy as np
from numpy import column_stack

from ConfusionMatrix import compute_tp_tn_fn_fp, compute_accuracy, compute_precision, compute_recall, compute_f1_score
from RandomForest import RandomForest
import mysql.connector
from mysql.connector import MySQLConnection, Error
# from ConnectDatabase import connectDatabase
from train_test_split import train_test_split

def connectDatabase():
    try:
        # Ket noi MySQL voi Python bang ham mysql.connector.connect()
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456",
            database="th3iot"
        )
        print("Ket noi thanh cong!")
    except:  # Truong hop co loi khi ket noi
        print("Kiem tra lai thong tin ket noi!")
    cur = db.cursor()
    cur.execute("SELECT * FROM test15k")
    df = pd.DataFrame(cur.fetchall())
    df.to_dict()
    return df, db




def insertEvaluateData(arr):
    # query = ("CREATE TABLE evaluatescratchRF (`id` INT, `acc` FLOAT, `precision_` FLOAT, `recall` FLOAT, `f1_score` FLOAT)")
    # sql = "INSERT INTO evaluatescratchRF (`id`, `acc`, `precision_`, `recall`, `f1_score`) VALUES(%s, %s, %s, %s, %s)"
    sql = "UPDATE evaluatescratchRF SET `acc` = %s, `precision_` = %s, `recall` = %s, `f1_score` = %s WHERE `id` = 1"
    try:
        # conn = connect()
        cursor = db.cursor()
        data = tuple(arr)
        print(data)
        # cursor.execute(query)
        cursor.execute(sql, data)
        print("Thanh cong")
        db.commit()
    except Error as error:
        print(error)
    finally:
        # Đóng kết nối
        cursor.close()
        db.close()
def train_test_split(df, train_percent,randomState):

    train = df.sample(frac=train_percent, random_state=randomState)
    x_train = train.drop(labels=['HeartDiseaseorAttack',],axis=1)
    y_train = train['HeartDiseaseorAttack']
    test = df.drop(train.index)
    x_test = test.drop(labels=['HeartDiseaseorAttack',],axis=1)
    y_test = test['HeartDiseaseorAttack']
    return x_train, x_test, y_train, y_test
# df = pd.read_csv("D:\\Hoc Ky 1_2022\\IOT\\AA\\1.csv")
# df, db = connectDatabase()
try:
    # Ket noi MySQL voi Python bang ham mysql.connector.connect()
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",
        database="th3iot"
    )
    print("Ket noi thanh cong!")
except: # Truong hop co loi khi ket noi
    print("Kiem tra lai thong tin ket noi!")

df = pd.read_sql("SELECT * FROM test15k",db)
df = df.head(15000)
# db.close()
pd.DataFrame(data=df)
# print(df)
X = df.drop(labels=['HeartDiseaseorAttack',],axis=1)
y = df['HeartDiseaseorAttack']



# print(df)

X_train, X_test, y_train, y_test = train_test_split(df, 0.7, 5)
print(X_train, "\n\n\n",X_test, "\n\n\n",y_train, "\n\n\n",y_test)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(X_train, "\n\n\n",X_test, "\n\n\n",y_train, "\n\n\n",y_test)


clf = RandomForest(n_trees=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
result = pd.DataFrame({'Actual': y_test, 'Predict': predictions})

# acc = accuracy(y_test, predictions)
# print(acc)

tp_rf, tn_rf, fp_rf, fn_rf = compute_tp_tn_fn_fp(y_test, predictions)
# print('TP for Random Forest :', tp_rf)
# print('TN for Random Forest :', tn_rf)
# print('FP for Random Forest :', fp_rf)
# print('FN for Random Forest :', fn_rf)

acc = compute_accuracy(tp_rf, tn_rf, fn_rf, fp_rf)
precision_ = compute_precision(tp_rf, fp_rf)
recall = compute_recall(tp_rf, fn_rf)
f1_score = compute_f1_score(y_test, predictions)
print('Accuracy for Random Forest :', acc)
print('Precision for Random Forest :', precision_)
print('Recall for Random Forest :', recall)
print('F1 score for Random Forest :', f1_score)
arr=[acc, precision_, recall, f1_score]
insertEvaluateData(arr)