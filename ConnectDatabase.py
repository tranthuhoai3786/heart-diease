import mysql.connector
import pandas as pd
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