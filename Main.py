#coding=utf8
import Train, MS, DB

# 训练模型
Train.training(MS.MODEL_GaussianNaiveBayes, 0.70)

# 从硬盘中获取训练好的分类器
# clf = Train.getCLF(MS.MODEL_GaussianNaiveBayes)

# 输出id为x的前n个关键词
# Train.getTopProbability(clf, 3, 41189)

# 将所有文本的关键词输出到txt与mysql
# Train.outputALlDianPingKeyWords(clf, 3)