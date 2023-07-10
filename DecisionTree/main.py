import pandas as pd
import numpy as np

from ConfusionMatrix import compute_tp_tn_fn_fp, compute_accuracy, compute_precision, \
    compute_recall, compute_f1_score
from InsertEvaluateDataDT import insertEvaluateData

from train_test_split import train_test_split
from mysql.connector import  Error
# # from ConnectDatabase import connectDatabase
from DecisionTree import Node
import mysql.connector
import pandas as pd



def insertEvaluateData(arr,db):
    # query = "CREATE TABLE evaluatescratch (`id` INT, `acc` FLOAT, `precision_` FLOAT, `recall` FLOAT, `f1_score` FLOAT)"
    # sql = "INSERT INTO evaluatescratch (`id`, `acc`, `precision_`, `recall`, `f1_score`) VALUES(%s, %s, %s, %s, %s)"
    sql = "UPDATE evaluatescratch SET `acc` = %s, `precision_` = %s, `recall` = %s, `f1_score` = %s WHERE `id` = 1"
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
    cur.execute("SELECT * FROM winequatity")
    df = pd.DataFrame(cur.fetchall())
    df.to_dict()
    return df, db
df,db = connectDatabase()

cursor = db.cursor()

# # db.close()








# df = df.head(15000)
# print(df)




#Chia tập train test
df = df.drop(columns={12,9,11,13,10,3,4,15})
X_train, X_test, y_train, y_test = train_test_split(df,0.7,5)



Y = y_train
X = X_train
features = list(X.columns)
# print(features)
hp = {
    'max_depth': 5,
    'min_samples_split': 200
}
#Khởi tạo node
root = Node(Y, X, **hp)
#Split tốt nhất tajo caay quyeets định
root.grow_tree()
#Print thông tin cây
root.print_tree()
#Dự đoán
results = X.copy()
predict = root.predict(X_test)

# print(predict)



result = pd.DataFrame({'Actual:': y_test, 'Predict:': predict})
print(result)

y_test = y_test.tolist()
y_test = np.array(y_test)
predict = np.array(predict)
# print(type(y_test), type(predict))
# predict = np.array()



tp_rf, tn_rf, fp_rf, fn_rf = compute_tp_tn_fn_fp(y_test, predict)
print('TP for Decision Tree :', tp_rf)
print('TN for Decision Tree :', tn_rf)
print('FP for Decision Tree :', fp_rf)
print('FN for Decision Tree :', fn_rf)

acc = compute_accuracy(tp_rf, tn_rf, fn_rf, fp_rf)
precision_ = compute_precision(tp_rf, fp_rf)
recall = compute_recall(tp_rf, fn_rf)
f1_score = compute_f1_score(y_test, predict)
print('Accuracy for Decision Tree :', acc)
print('Precision for Decision Tree :', precision_)
print('Recall for Decision Tree :', recall)
print('F1 score for Decision Tree :', f1_score)
arr = [ acc, precision_, recall, f1_score]


insertEvaluateData(arr,db)