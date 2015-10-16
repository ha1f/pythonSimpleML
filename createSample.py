#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def randomPos(centerX,centerY):
    func = gaus(1,0)
    x=func(*1)
    y=func(random.random()*1)
    return (x,y)


def gaus(sigma, mu):
    return lambda x: 1.0 / math.sqrt(2.0 * math.pi * sigma*sigma) * math.exp(-(x-mu)*(x-mu)/(2*sigma*sigma))

def xArray(minVal, maxVal, dx):
    return [minVal + dx*i for i in range(0, int((maxVal-minVal)/dx)+1)]

def main():
    print("start")
    fig = plt.figure()

    sampleCount = 200

    xs = np.random.randn(sampleCount)
    ys = np.random.randn(sampleCount)
    data = [[x,y,-1] for x,y in zip(xs,ys)]
    plt.plot(xs,ys, "ro")

    xs = [np.random.randn()+4 for i in range(0,sampleCount)]
    ys = [np.random.randn()+2 for i in range(0,sampleCount)]
    data.extend([[x,y,1] for x,y in zip(xs,ys)])
    plt.plot(xs,ys, "go")

    plt.show()

    #output
    f = open("sampleData.data", 'w')
    f.write('[\n')
    for datum in data:
        f.write(str(datum)+',\n')
    f.write(']\n')
    f.close()

main()
