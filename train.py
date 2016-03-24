#coding=utf8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import numpy as np
from sklearn import naive_bayes, metrics, preprocessing
from sklearn.externals import joblib
import os
import Data, MS

# 计算分类器的准确率和召回率（常规）
def calculate_result_normal(actual,pred):
    m_precision = metrics.precision_score(actual,pred);
    m_recall = metrics.recall_score(actual,pred);
    print 'predict info:'
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall);
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred));

# 计算分类器的准确率和召回率（关键词抽取-每篇取前三个）
def calculate_result_keyword():
    pass

# 训练模型并保存到硬盘，其中model为分类器类型，size为训练集占比（如0.75）
def training(model, size):
    tt = Data.getData();
    # 训练集和测试集
    SIZE = int(size * len(tt["data"]))
    trainX = tt["data"][:SIZE]
    trainY = tt["target"][:SIZE]
    testX = tt["data"][SIZE:]
    testY = tt["target"][SIZE:]
    # 数据标准化
    scaleX = preprocessing.StandardScaler().fit(trainX)
    trainX = scaleX.transform(trainX)
    testX = scaleX.transform(testX)
    # 数据正则化
    # scaleX = preprocessing.Normalizer().fit(trainX)
    # trainX = scaleX.transform(trainX)
    # testX = scaleX.transform(testX)
    # 数据归一化
    # scaleX = preprocessing.MinMaxScaler().fit(trainX)
    # trainX = scaleX.transform(trainX)
    # testX = scaleX.transform(testX)
    # 训练模型
    if model == MS.MODEL_GaussianNaiveBayes:
        clf = naive_bayes.GaussianNB()
        clf.fit(trainX, trainY)
        pred = clf.predict(testX)
    calculate_result_normal(testY, pred)
    # 将模型保存到硬盘
    path = "E:\\work\\迪士尼\\output\\" + str(model) + ".pkl"
    joblib.dump(clf, path.decode("utf8").encode("gb2312"), compress=3)
    print "已将模型(编号:" + str(model) + ")保存到" + path

# 读取硬盘中训练好的模型
def getCLF(model):
    path = "E:\\work\\迪士尼\\output\\" + str(model) + ".pkl"
    clf = joblib.load(path.decode("utf8").encode("gb2312"))
    return clf

# 根据分类器给出的“是关键词”可能性排序，输出前n个
# 第一个参数表示使用的训练好的分类器，第二个参数表示输出topN，第三个参数表示文本id号
def getTopProbability(clf, n, id):
    pathX = ("E:\\work\\迪士尼\\output\\单篇特征值\\" + str(id) + ".txt").decode("utf8").encode("gb2312")
    pathW = ("E:\\work\\迪士尼\\output\\单篇特征值所对应的词\\" + str(id) + ".txt").decode("utf8").encode("gb2312")
    # 获取特征值
    fX = open(pathX)
    oneX = []
    for line in fX.readlines():
        # 一开始获取的是string类型的数字
        rawX = line.replace("\n", "").split(",")
        # 转换成浮点型
        x = []
        for f in rawX:
            x.append(float(f))
        oneX.append(x)
    fX.close()
    # 获取特征词
    fW = open(pathW)
    oneW = []
    for line in fW.readlines():
        rawW = line.replace("\n", "")
        oneW.append(rawW)
    fW.close()
    # 将特征词和关键词可能性关联
    pKeyWord = {}
    for i in range(len(oneX)):
        pKeyWord[oneW[i]] = clf.predict_proba(np.array(oneX[i]).reshape(-1, 1))[0][1]
    sortPKW = sorted(pKeyWord.iteritems(), key=lambda d:d[1], reverse=True)
    print sortPKW
    result = []
    pResult = ""
    for i in (range(n) if n < len(sortPKW) else range(len(sortPKW))):
        try:
            pResult += sortPKW[i][0].decode("gbk") + ","
            result.append(sortPKW[i][0].decode("gbk"))
        except:
            pass
    print pResult[:-1]
    return result

def outputALlDianPingKeyWords(clf, n):
    list = []
    for s in os.listdir("E:\\work\\迪士尼\\output\\单篇特征值".decode("utf8").encode("gb2312")):
        list.append(getTopProbability(clf, n, s[:-4]))
    print list
    fw = open("E:\\work\\迪士尼\\output\\keywords.txt".decode("utf8").encode("gb2312"), "w")
    for words in list:
        line = ""
        for word in words:
            line += word + ","
        fw.write(line[:-1] + "\r\n")
    fw.close()
    print "已将所有点评的关键词输出到E:\\work\\迪士尼\\output\\keywords.txt"