#coding=utf8
import Train, MS, DB

# 训练模型
# 第一个参数表示分类器，第二个参数表示训练集的大小,第三个参数表示至多选取n个关键词，第四个参数表示判定为关键词的最小阈值
Train.training(MS.MODEL_TreeClassifier, 0.70, 5, 0.6)

# 从硬盘中获取训练好的分类器
# clf = Train.getCLF(MS.MODEL_TreeClassifier)

# 输出id为x的至多前n个关键词
#第一个参数表示分类器，第二个参数表示最多几个关键词，第三个参数表示判定为关键词的最小阈值，第四个参数为id
# Train.getTopProbability(clf, 5, 0.5, 106868)

# 将所有文本的关键词输出到txt与mysql
# Train.outputALlDianPingKeyWords(clf, 5, 0.6)