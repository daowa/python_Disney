import MySQLdb

def getData():
    try:
        conn=MySQLdb.connect(host='localhost',user='root',passwd='abcd@123',db='disney',port=3306,charset='utf8')
        cur=conn.cursor()
        cur.execute('select * from training_set')
        result = cur.fetchall()
        print result[1][2]
        cur.close()
        conn.close()
    except MySQLdb.Error,e:
         print "Mysql Error %d: %s" % (e.args[0], e.args[1])