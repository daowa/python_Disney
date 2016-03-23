#coding=utf8
import Train, MS

# 训练模型
Train.training(MS.MODEL_GaussianNaiveBayes, 0.75)

# 从硬盘中获取训练好的分类器
# clf = Train.getCLF(MS.MODEL_GaussianNaiveBayes)

# 输出前n
# Train.getTopProbability(clf, 4)

