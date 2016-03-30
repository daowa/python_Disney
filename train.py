#coding=utf8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import numpy as np
from sklearn import naive_bayes, metrics, preprocessing
from sklearn.externals import joblib
import os
import Data, MS, DB

# 计算分类器的准确率和召回率（常规）
def calculate_result_normal(actual, pred):
    print "#######################"
    m_precision = metrics.precision_score(actual,pred);
    m_recall = metrics.recall_score(actual,pred);
    print 'predict info(normal):'
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall)
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred))
    print "#######################"

# 计算分类器的准确率和召回率（关键词抽取-每篇取前三个）
def calculate_result_keyword(tt, clf, size):
    # 计算提取出正确的词数量、总共提取出的词数、样本中的词数
    print "#######################"
    SIZE = int(size * len(tt["id"]))
    ids = tt["id"][SIZE:]
    keywords = tt["keywords"][SIZE:]
    topTFIDFs = tt["topTFIDF"][SIZE:]
    topFrequencys = tt["topFrequency"][SIZE:]
    # 正确的词数量
    countTrue = 0
    countTrue_tfidf = 0
    countTrue_frequency = 0
    # 总共提取出的词数
    countPredict = 0
    # 样本中的词数
    countAll = len(tt["data"][int(size * len(tt["data"])):])
    for i in range(len(keywords)):
        predictWords = getTopProbability(clf, 3, ids[i])
        countPredict += len(predictWords)
        for j in range(len(predictWords)):#分类器输出的关键词
            if predictWords[j] in keywords[i]:
                countTrue += 1
        for j in range(len(topTFIDFs[i])):#topTFIDFs输出的关键词
            if topTFIDFs[i][j] in keywords[i]:
                countTrue_tfidf += 1
        for j in range(len(topFrequencys[i])):#topFrequencys输出的关键词
            if topFrequencys[i][j] in keywords[i]:
                countTrue_frequency += 1
    #计算准确率(有空改成个函数计算吧，现在下面写那么多变量好难看）
    precision = float(countTrue)/float(countPredict)
    precision_tfidf = float(countTrue_tfidf)/float(countPredict)
    precision_frequency = float(countTrue_frequency)/float(countPredict)
    #计算召回率
    recall = float(countTrue)/float(countAll)
    recall_tfidf = float(countTrue_tfidf)/float(countAll)
    recall_frequency = float(countTrue_frequency)/float(countAll)
    #计算F值
    f1 = precision*recall*2 / float(precision + recall)
    f1_tfidf = precision_tfidf*recall_tfidf*2 / float(precision_tfidf + recall_tfidf)
    f1_frequency = precision_frequency*recall_frequency*2 / float(precision_frequency + recall_frequency)
    print 'predict info(keywords):'
    print 'precision:{0:.3f}'.format(precision) + " (tfidf:{0:.3f}".format(precision_tfidf) + ")(frequency:{0:0.3f}".format(precision_frequency) + ")"
    print 'recall:{0:0.3f}'.format(recall) + " (tfidf:{0:.3f}".format(recall_tfidf) + ")(frequency:{0:0.3f}".format(recall_frequency) + ")"
    print 'f1-score:{0:.3f}'.format(f1) + " (tfidf:{0:.3f}".format(f1_tfidf) + ")(frequency:{0:0.3f}".format(f1_frequency) + ")"
    print "#######################"

# 训练模型并保存到硬盘，其中model为分类器类型，size为训练集占比（如0.75）
def training(model, size):
    tt = Data.getTrainingtData()
    # 训练集和测试集
    SIZE = int(size * len(tt["data"]))
    trainX = tt["data"][:SIZE]
    trainY = tt["target"][:SIZE]
    testX = tt["data"][SIZE:]
    testY = tt["target"][SIZE:]
    # 数据标准化
    # scaleX = preprocessing.StandardScaler().fit(trainX)
    # trainX = scaleX.transform(trainX)
    # testX = scaleX.transform(testX)
    # 数据正则化
    # scaleX = preprocessing.Normalizer().fit(trainX)
    # trainX = scaleX.transform(trainX)
    # testX = scaleX.transform(testX)
    # 数据归一化
    # scaleX = preprocessing.MinMaxScaler().fit(trainX)
    # trainX = scaleX.transform(trainX)
    # testX = scaleX.transform(testX)
    print trainX
    # 训练模型
    if model == MS.MODEL_GaussianNaiveBayes:
        clf = naive_bayes.GaussianNB()
    elif model == MS.MODEL_MultinomialNaiveBayes:
        clf = naive_bayes.MultinomialNB(alpha=0.01)
    elif model == MS.MODEL_BernoulliNaiveBayes:
        clf = naive_bayes.BernoulliNB()
    clf.fit(trainX, trainY)
    # 评估模型效果
    pred = clf.predict(testX)
    print pred
    calculate_result_normal(testY, pred)
    calculate_result_keyword(tt, clf, size)
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
        pKeyWord[oneW[i]] = clf.predict_proba(np.array(oneX[i]).reshape(1, -1))[0][1]
    sortPKW = sorted(pKeyWord.iteritems(), key=lambda d:d[1], reverse=True)
    # print sortPKW
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
    dict = {}
    for s in os.listdir("E:\\work\\迪士尼\\output\\单篇特征值".decode("utf8").encode("gb2312")):
        dict[s[:-4]] = getTopProbability(clf, n, s[:-4])
    # 输出到txt
    fw = open("E:\\work\\迪士尼\\output\\keywords.txt".decode("utf8").encode("gb2312"), "w")
    for (k,v) in dict.items():
        line = ""
        for word in v:
            line += word + ","
        fw.write(line[:-1] + "\r\n")
    fw.close()
    print "已将所有点评的关键词输出到E:\\work\\迪士尼\\output\\keywords.txt"
    # 输出到mysql
    print "开始输入数据库"
    DB.intsertKeyWords(dict)
    print "已将所有点评的关键词输出到MySql"
    print "done"