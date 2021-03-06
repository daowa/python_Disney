#coding=utf8
from numpy import *

# 导入训练数据
def getTrainingtData():
    # 获取data
    data = []
    path = "E:\\work\\迪士尼\\output\\trainingX.txt".decode("utf8").encode("gb2312")
    fX = open(path)
    for line in fX.readlines():
        # 一开始获取的是string类型的数字
        rawX = line.replace("\n", "").split(",")
        # 转换成浮点型
        x = []
        for f in rawX:
            x.append(float(f))
        data.append(x)
    fX.close()
    # 获取target
    target = []
    path = "E:\\work\\迪士尼\\output\\trainingY.txt".decode("utf8").encode("gb2312")
    fY = open(path)
    for line in fY.readlines():
        y = int(line.replace("\n", ""))
        target.append(y)
    fY.close()
    # 获取id
    ids = []
    path = "E:\\work\\迪士尼\\output\\trainingID.txt".decode("utf8").encode("gb2312")
    fID = open(path)
    for line in fID.readlines():
        line = line.replace("\n", "")#(ATTENTION!)去换行
        ids.append(int(line))
    fID.close()
    # 获取关键词
    keywords = []
    path = "E:\\work\\迪士尼\\output\\trainingWord.txt".decode("utf8").encode("gb2312")
    fKW = open(path)
    for line in fKW.readlines():
        line = line.replace("\n", "")#(ATTENTION!)去换行
        temp = line.split(",")
        kws = []
        for i in range(len(temp)):
            kws.append(temp[i].decode("gbk"))
        keywords.append(kws)
    fKW.close()
    # 获取topTFIDF的词
    topTFIDFS = []
    path = "E:\\work\\迪士尼\\output\\trainingTopTFIDF.txt".decode("utf8").encode("gb2312")
    fTI = open(path)
    for line in fTI.readlines():
        line = line.replace("\n", "")#(ATTENTION!)去换行
        temp = line.split(",")
        tfidf = []
        for i in range(len(temp)):
            tfidf.append(temp[i].decode("gbk"))
        topTFIDFS.append(tfidf)
    fTI.close()
    # 获取top词频的词
    topFrequencys = []
    path = "E:\\work\\迪士尼\\output\\trainingTopFrequency.txt".decode("utf8").encode("gb2312")
    fFQ = open(path)
    for line in fFQ.readlines():
        line = line.replace("\n", "")#(ATTENTION!)去换行
        temp = line.split(",")
        frequency = []
        for i in range(len(temp)):
            frequency.append(temp[i].decode("gbk"))
        topFrequencys.append(frequency)
    fFQ.close()
    # 放入字典中，之后可以根据下标查询到同一个样本的 id、特征、目标、关键词(并不能现在。。只看前半句吧)
    tt = {"data":data, "target":target, "id":ids, "keywords": keywords, "topTFIDF": topTFIDFS, "topFrequency": topFrequencys}
    return tt

