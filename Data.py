#coding=utf8
from numpy import *

# 导入数据
def getData():
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