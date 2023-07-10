import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import mysql.connector
import pandas as pd
from mysql.connector import MySQLConnection, Error
from ConnectDatabase import connectDatabase

def insertEvaluateData(arr):
    # query = ("CREATE TABLE evaluatefuncRF (`id` INT, `acc` FLOAT, `precision_` FLOAT, `recall` FLOAT, `f1` FLOAT)")
    # sql = "INSERT INTO evaluatefuncRF (`id`, `acc`, `precision_`, `recall`, `f1`) VALUES(%s, %s, %s, %s, %s)"
    sql = "UPDATE evaluatefuncRF SET `acc` = %s, `precision_` = %s, `recall` = %s, `f1` = %s WHERE `id` = 1"
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

# df = pd.read_csv("D:\\Hoc Ky 1_2022\\IOT\\AA\\test15k.csv")
df,db = connectDatabase()
df=df.head(15000)
X = df.drop(columns=0,axis=1)
y = df[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=5)

model = RandomForestClassifier(max_depth=10, n_estimators=10,min_samples_split=2)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
# ac = accuracy_score(y_test, y_predict)
# print(ac)


tn_rf1, fp_rf1, fn_rf1, tp_rf1 = confusion_matrix(y_test, y_predict).ravel()
print('TP for Random Forest :', tp_rf1)
print('TN for Random Forest :', tn_rf1)
print('FP for Random Forest :', fp_rf1)
print('FN for Random Forest :', fn_rf1)

acc = 100*accuracy_score(y_test, y_predict)
precision_ = 100* precision_score(y_test, y_predict)
recall = 100* recall_score(y_test, y_predict)
f1 = 100*f1_score(y_test, y_predict)
print('Accuracy for Random Forest :', acc)
print('Precision for Random Forest :', precision_)
print('Recall for Random Forest :', recall)
print('F1 score for Random Forest :', f1)
arr = [acc, precision_, recall, f1]
insertEvaluateData(arr)