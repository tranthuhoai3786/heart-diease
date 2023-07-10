import mysql.connector
import pandas as pd
import seaborn as sns
import math
from mysql.connector import MySQLConnection, Error
from ConnectDatabase import connectDatabase
from correlation import cor
import numpy as np
# hàm tính correlation bằng tay
# def cor(x, y):
#     # Finding the mean of the series x and y
#     mean_x = sum(x) / float(len(x))
#     mean_y = sum(y) / float(len(y))
#     # Subtracting mean from the individual elements
#     sub_x = [i - mean_x for i in x]
#     sub_y = [i - mean_y for i in y]
#     numerator = sum([sub_x[i] * sub_y[i] for i in range(len(sub_x))])
#     denominator = len(x) - 1
#     cov = numerator / denominator
#
#     covx = math.sqrt(sum(i ** 2 for i in sub_x))
#     covy = math.sqrt(sum(i ** 2 for i in sub_y))
#     # print(covx)
#     # print(covy)
#     result = numerator / (covx * covy)
#     return result


def insertCorrData(data):
    query1 = "CREATE TABLE correlation (`HeartDiseaseorAttack` FLOAT, `HighBP` FLOAT, `HighChol` FLOAT, `CholCheck` FLOAT, `BMI` FLOAT, `Smoker` FLOAT, `Stroke` FLOAT, `Diabetes` FLOAT, `PhysActivity` FLOAT, `Fruits` FLOAT, `Veggies` FLOAT, `HvyAlcoholConsump` FLOAT, `AnyHealthcare` FLOAT, `NoDocbcCost` FLOAT, `GenHlth` FLOAT, `MentHlth` FLOAT, `PhysHlth` FLOAT, `DiffWalk` FLOAT, `Sex` FLOAT, `Age` FLOAT,`Education` FLOAT, `Income` FLOAT)"
    # sql = "INSERT INTO evaluatescratch (`id`, `acc`, `precision_`, `recall`, `f1_score`) VALUES(%s, %s, %s, %s, %s)"
    query = "INSERT INTO correlation(HeartDiseaseorAttack, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, Diabetes, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income) VALUES(%s,%s,%s,%s, %s,%s, %s,%s, %s,%s, %s,%s, %s,%s, %s,%s, %s,%s, %s,%s, %s, %s)"
    try:
        # conn = connect()
        cursor = db.cursor()
        data = tuple(data)
        print(data)
        cursor.execute(query1)
        cursor.execute(query, data)
        print("Thanh cong")
        db.commit()
    except Error as error:
        print(error)
    finally:
        # Đóng kết nối
        cursor.close()
        db.close()

df,db = connectDatabase()


#kiem tra du lieu k phu hop
print(df.isnull().sum())


df1 = df.copy()

corr=[]
for i in df1:
    x = df1[i].tolist()
    y = df1[0].tolist()
    correlation = cor(x, y)
    corr.append(correlation)

print(corr)
# print("This is Correlation: \n\n\n", corr)
# data = np.array(corr[0])

insertCorrData(corr)

#
# df,db = connectDatabase()
# df = df.head(15000)
# #kiem tra gia tri null trong data
#
# isna = df.isna().sum()
# print(isna)
#
# #----khong co gia tri nao bi null---
# #ve heatmap the hien do tuong quan cua cac dac trung
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# plt.figure(figsize=(10,15))  # on this line I just set the size of figure to 12 by 10.,
# p=sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')
# plt.show()
#
# #tính độ tương quan giữa các đặc trưng
# df1 = df.copy()
#
#
# for i in df1:
#     x = df1[i].tolist()
#     y = df1[0].tolist()
#     correlation = cor(x, y)
#     print(correlation)

#được bảng correlation
#các đặc trưng có độ tương quan cao nhất là

#14 (0.26),19(0.22), 17,1(0.21), 6,(0.2), 2,7,16 (0.18), 21(0.14), 5(0.11), 20(0.1)
#8(0.087), 18(0.086), 15(0.065), 4(0.053), 3(0.044), 10(0.039), 13(0.031), 11(0.029), 9(0.02), 12(0.019),