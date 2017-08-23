import MySQLdb

conn = MySQLdb.Connect(host = '127.0.0.1',
                       port = 3306,
                       user = 'root',
                       passwd = 'Zf12130827',
                       db = 'test'
                        )

cursor = conn.cursor()

print conn
print cursor

cursor.close()
conn.close()