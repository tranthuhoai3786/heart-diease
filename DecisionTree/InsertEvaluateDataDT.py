from mysql.connector import  Error
# import mysql.connector
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