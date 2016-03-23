#coding=utf8
from numpy import *
from sklearn import datasets, svm, naive_bayes, metrics
from sklearn.externals import joblib
import Data, MS

# 计算分类器的准确率和召回率（常规）
def calculate_result(actual,pred):
    m_precision = metrics.precision_score(actual,pred);
    m_recall = metrics.recall_score(actual,pred);
    print 'predict info:'
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall);
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred));

# 训练模型并保存到硬盘，其中model为分类器类型，size为训练集占比（如0.75）
def training(model, size):
    tt = Data.getData();
    # 训练集和测试集
    SIZE = int(size * len(tt["data"]))
    trainX = tt["data"][:SIZE]
    trainY = tt["target"][:SIZE]
    testX = tt["data"][SIZE:]
    testY = tt["target"][SIZE:]
    # 训练模型
    if model == MS.MODEL_GaussianNaiveBayes:
        clf = naive_bayes.GaussianNB()
        clf.fit(trainX, trainY)
        pred = clf.predict(testX)
    calculate_result(testY, pred)
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
# 第一个参数表示使用的训练好的分类器，第二个参数表示输出topN
def getTopProbability(clf, n):
    pathX = "E:\\work\\迪士尼\\output\\单篇特征值\\35664.txt".decode("utf8").encode("gb2312")
    pathW = "E:\\work\\迪士尼\\output\\单篇特征值所对应的词\\35664.txt".decode("utf8").encode("gb2312")
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
        pKeyWord[oneW[i]] = clf.predict_proba(oneX[i])[0][1]
    print pKeyWord
    sortPKW = sorted(pKeyWord.iteritems(), key=lambda d:d[1], reverse=True)
    print sortPKW
    for i in range(n):
        print(sortPKW[i][0].decode("gb2312"))