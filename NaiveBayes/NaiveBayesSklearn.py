
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from mysql.connector import  Error
from ConnectDatabase import connectDatabase

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def insertEvaluateData(arr):
    # query = ("CREATE TABLE evaluatefuncNB (`id` INT, `acc` FLOAT, `precision_` FLOAT, `recall` FLOAT, `f1` FLOAT)")
    sql = "INSERT INTO evaluatefuncNB (`id`, `acc`, `precision_`, `recall`, `f1`) VALUES(%s, %s, %s, %s, %s)"
    # sql = "UPDATE evaluatefuncNB SET `acc` = %s, `precision_` = %s, `recall` = %s, `f1` = %s WHERE `id` = 1"
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


df,db = connectDatabase()
df=df.head(15000)
X = df.drop(columns=0, axis=1)
y = df[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

model = GaussianNB()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
# ac = accuracy_score(y_test, y_predict)
# print(ac)


tn_rf1, fp_rf1, fn_rf1, tp_rf1 = confusion_matrix(y_test, y_predict).ravel()
print('TP for Bayes :', tp_rf1)
print('TN for Bayes :', tn_rf1)
print('FP for Bayes :', fp_rf1)
print('FN for Bayes :', fn_rf1)

acc = 100*accuracy_score(y_test, y_predict)
precision_ = 100* precision_score(y_test, y_predict)
recall = 100* recall_score(y_test, y_predict)
f1 = 100*f1_score(y_test, y_predict)
print('Accuracy for Bayes :', acc)
print('Precision for Bayes :', precision_)
print('Recall for Bayes :', recall)
print('F1 score for Bayes :', f1)
arr = [1, acc, precision_, recall, f1]
insertEvaluateData(arr)