#coding=utf8
from numpy import *
from sklearn import datasets, svm, naive_bayes, metrics
from string import *

# 导入数据
def initData():
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
    tt = {"data":data, "target":target}
    return tt

def calculate_result(actual,pred):
    m_precision = metrics.precision_score(actual,pred);
    m_recall = metrics.recall_score(actual,pred);
    print 'predict info:'
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall);
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred));

tt = initData();
# 训练集和测试集
SIZE = int(0.75 * len(tt["data"]))
trainX = tt["data"][:SIZE]
trainY = tt["target"][:SIZE]
testX = tt["data"][SIZE:]
testY = tt["target"][SIZE:]
# 训练模型
clf = naive_bayes.GaussianNB()
clf.fit(trainX, trainY)
pred = clf.predict(testX)
print pred
print testY
calculate_result(testY, pred)

# 根据判断的可能性输出前三个词
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
for i in range(3):
    print(sortPKW[i][0].decode("gb2312"))

# iris = datasets.load_iris()
# digits = datasets.load_digits()

# print(iris.data)
# print(iris.target)
# print (digits.data)

# print(digits)
# print(digits.images[0])
# print(iris)

# clf = svm.SVC(gamma = 0.001, C = 100)
# clf.fit(digits.data[:-1], digits.target[:-1])
# print(clf)
#
# print(clf.predict(digits.data[-1]))

# print(iris)
# print(iris.DESCR)

# print iris.data
# print digits

# a = np.matrix('1 2 7; 3 4 8; 5 6 9')

