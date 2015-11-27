#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Util:
    # dx間隔の数列を生成
    numArray = staticmethod(lambda minVal, maxVal, dx: [minVal + dx*i for i in range(0, int((maxVal-minVal)/dx)+1)])

class Perceptron:
    # out = ax + by + c
    params=[]

    # 出力の重み関数、今回は不使用
    weight = staticmethod(lambda x: x)

    # 学習係数
    K = 0.01

    def __init__(self, params):
        self.params = params

    def getOutput(self, inputData):
        return Perceptron.weight(sum([i*j for i,j in zip(inputData, self.params)]))

    def learn(self, teacherIn, teacherOut):
        diff0 = teacherOut - self.getOutput(teacherIn)
        diff = (1 if diff0>0 else -1) * pow(diff0, 2)
        # 間違った時のみ更新
        if abs(diff) > 1:
            for (i, inDataI) in enumerate(teacherIn):
                self.params[i] += self.K * diff * inDataI

    # 現在の境界をplot
    def plotBoundary(self, style, width):
        xs = Util.numArray(-5,50,0.1)
        #y = -a/b*x - c/bから、それぞれのxについてyを求める
        ys = [(- self.params[0]*x - self.params[2]*1)/self.params[1] for x in xs]
        plt.plot(xs, ys, style,linewidth=width)

def loadTeacherData(fileName):
    teacherData = []
    f = open(fileName, 'r')
    for line in f:
        data = line.replace("[", "").replace("],", "").split(", ")
        if len(data) > 1:
            teacherData.append((map(float, data[0:-1])+[1], float(data[-1])))
    return teacherData

def main():
    teacherDataTmp = loadTeacherData('sampleData.data')
    #teacherData = [ [[math.sqrt(pow(data[0][0],2) + pow(data[0][1],2)), math.atan2(data[0][1], data[0][0]), data[0][2]] , data[1]] for data in teacherDataTmp]
    teacherData = teacherDataTmp

    random.shuffle(teacherData)

    # パラメータの初期化
    perceptron = Perceptron([random.uniform(-0.1,0.1)]*(len(teacherData[0][0])))

    # 学習フェーズ
    for epoc in range(0,100):
        # 現在の境界をプロット
        perceptron.plotBoundary("y-", 1)
        # 学習
        for teacherIn, teacherOut in teacherData:
            perceptron.learn(teacherIn, teacherOut)

    # 確認フェーズ
    cCount = 0
    for teacherIn, teacherOut in teacherData:
        # 点をプロット
        plt.plot(teacherIn[0],teacherIn[1], ("ro" if teacherOut < 0 else "go"))
        # その点の入力に対する出力が等しいかどうか
        res = 1 if perceptron.getOutput(teacherIn) > 0 else -1
        cCount += 1 if teacherOut == res else 0
    dataCount = len(teacherData)
    print("accuracy:" +str(cCount) + "/" + str(dataCount) + " = " + str(100.0 * cCount/dataCount) + "%")

    # 結果をプロット
    perceptron.plotBoundary("b-",3)

    plt.xlim(-4, 8)
    plt.ylim(-3, 5)
    plt.show()

main()
