#coding=utf8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import numpy as np
from sklearn import naive_bayes, svm, tree, neighbors, metrics, preprocessing, cross_validation
from sklearn.externals import joblib
import random
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
def calculate_result_keyword(tt, clf, size, topN, threshold):
    # 计算提取出正确的词数量、总共提取出的词数、样本中的词数
    print "#######################"
    SIZE = int(size * len(tt["id"]))
    ids = tt["id"][SIZE:]
    keywords = tt["keywords"][SIZE:]
    topTFIDFs = tt["topTFIDF"][SIZE:]
    topFrequencys = tt["topFrequency"][SIZE:]
    # 正确的词数量
    countTrue = 0 #关键词抽取
    countTrue_tfidf = 0 #tfidf
    countTrue_frequency = 0 #词频
    # 预测的正例数
    countPredict = 0
    # 实际的正例数
    countPos = 0
    for i in range(len(keywords)):
        predictWords = getTopProbability(clf, topN, threshold, ids[i])
        countPredict += len(predictWords)
        countPos += len(keywords[i])
        for j in range(len(predictWords)):#分类器输出的关键词
            if predictWords[j] in keywords[i]:
                countTrue += 1
        for j in range(len(topTFIDFs[i])):#topTFIDFs输出的关键词
            if topTFIDFs[i][j] in keywords[i]:
                countTrue_tfidf += 1
        for j in range(len(topFrequencys[i])):#topFrequencys输出的关键词
            if topFrequencys[i][j] in keywords[i]:
                countTrue_frequency += 1
    print 'predict info(keywords):'
    print 'precision:{0:.3f}'.format(getPRF(countTrue, countPredict, countPos)[0]) + " (tfidf:{0:.3f}".format(getPRF(countTrue_tfidf, countPredict, countPos)[0]) + ")(frequency:{0:0.3f}".format(getPRF(countTrue_frequency, countPredict, countPos)[0]) + ")"
    print 'recall:{0:0.3f}'.format(getPRF(countTrue, countPredict, countPos)[1]) + " (tfidf:{0:.3f}".format(getPRF(countTrue_tfidf, countPredict, countPos)[1]) + ")(frequency:{0:0.3f}".format(getPRF(countTrue_frequency, countPredict, countPos)[1]) + ")"
    print 'f1-score:{0:.3f}'.format(getPRF(countTrue, countPredict, countPos)[2]) + " (tfidf:{0:.3f}".format(getPRF(countTrue_tfidf, countPredict, countPos)[2]) + ")(frequency:{0:0.3f}".format(getPRF(countTrue_frequency, countPredict, countPos)[2]) + ")"
    print "#######################"

def getPRF(countTrue, countPredict, countPos):
    precision = float(countTrue)/float(countPredict) #准确率：你预测的有多少是对的    P = 正确预测/预测的正例数
    recall = float(countTrue)/float(countPos) #召回率：正例里你覆盖了多少    R = 正确预测/实际的正例数
    f1 = precision*recall*2 / float(precision + recall)
    return (precision, recall, f1)

# 训练模型并保存到硬盘，其中model为分类器类型，size为训练集占比（如0.75）
def training(model, size, topN, threshold):
    tt = Data.getTrainingtData()
    # 训练集和测试集
    trainX, testX, trainY, testY = cross_validation.train_test_split(tt["data"], tt["target"], test_size=(1-size), random_state=None)
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
    elif model == MS.MODEL_SvmSVC:
        clf = svm.SVC(gamma=0.001, C=100, probability=True)
    elif model == MS.MODEL_SvmNuSVC:
        clf = svm.NuSVC(probability=True)
    elif model == MS.MODEL_TreeClassifier:
        clf = tree.DecisionTreeClassifier()
    elif model == MS.MODEL_KNeighbors:
        clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    clf.fit(trainX, trainY)
    # 评估模型效果
    pred = clf.predict(testX)
    print pred
    calculate_result_normal(testY, pred)
    calculate_result_keyword(tt, clf, size, topN, threshold)
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
def getTopProbability(clf, n, threshold, id):
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
    result = []
    pResult = ""
    for i in (range(n) if n < len(sortPKW) else range(len(sortPKW))):
        try:
            if sortPKW[i][1] < threshold: #为真的阈值，这里先采用大于反面即为真，后期可不断调
                break
            pResult += sortPKW[i][0].decode("gbk") + ","
            result.append(sortPKW[i][0].decode("gbk"))
        except:
            pass
    print pResult[:-1]
    return result

def outputALlDianPingKeyWords(clf, n, threshold):
    dict = {}
    for s in os.listdir("E:\\work\\迪士尼\\output\\单篇特征值".decode("utf8").encode("gb2312")):
        dict[s[:-4]] = getTopProbability(clf, n, threshold, s[:-4])
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