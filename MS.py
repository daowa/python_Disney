#coding=utf8
# 该类用于存储key等全局变量

MODEL_GaussianNaiveBayes = 0;#适用于高斯分布（正态分布）的特征
MODEL_MultinomialNaiveBayes = 1;#这个分类器以出现次数作为特征值，我们使用的TF-IDF也能符合这类分布
MODEL_BernoulliNaiveBayes = 2;#适用于伯努利分布（二值分布）的特征