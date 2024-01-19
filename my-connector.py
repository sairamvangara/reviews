import mysql.connector

mydb=mysql.connector.connect(
    host='localhost',
    user='root',
    password='Devika@19',
    port='3306',
    database='userlogin'
)
mycursor=mydb.cursor()
mycursor.execute('SELECT *FROM user')
users = mycursor.fetchall()
for user in users:
    print(user)