import MySQLdb

conn=MySQLdb.connect(host='localhost',user='root',passwd='abcd@123',db='disney',port=3306,charset='utf8')
cur=conn.cursor()

def init():
    conn=MySQLdb.connect(host='localhost',user='root',passwd='abcd@123',db='disney',port=3306,charset='utf8')
    cur=conn.cursor()

def close():
    cur.close()
    conn.close()

def getData():
    try:
        init()
        cur.execute('select * from training_set')
        result = cur.fetchall()
        print result[1][2]
        close()
    except MySQLdb.Error,e:
         print "Mysql Error %d: %s" % (e.args[0], e.args[1])

def intsertKeyWords(dict):
    try:
        init()
        for(k,v) in dict.items():
            keywords = ""
            for word in v:
                keywords += word + ","
            keywords = keywords[:-1]
            cur.execute("insert into keywords(idraw_dianping, keywords) values (" + k + ", \"" + keywords + "\")")
        conn.commit()
        close()
    except MySQLdb.Error,e:
         print "Mysql Error %d: %s" % (e.args[0], e.args[1])