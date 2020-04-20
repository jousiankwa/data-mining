import numpy as np
import math
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']


def loadData(fileName):
    x = [[], [], [], [], [], [], [], [], [], []]
    inFile = open(fileName, 'r')  # 以只读方式打开fileName文件
    for c in inFile.readlines():
        c_array=c.split(',')
        for i in range(10):
            x[i].append(float(c_array[i]))
    return x

a=loadData("magic04.txt")
x = np.array(a) #将原始数据化成矩阵
z = np.array(x.mean(axis=1)).reshape(10, 1) #计算原数据的均值向量（10*1）
p=x-z   #计算中心化后的矩阵
def inner():
    #使用内积计算协方差矩阵，存储在y中
    y=np.dot(x-z,(x-z).T)/19019
    return y

def outer():
    #使用外积计算协方差矩阵，最后存储在d中，为10*10矩阵
    d=np.zeros(shape=(10,10))
    for c in range(19020):
        b=p.T[c]
        d=d+np.outer(b,b)
    d=d/19019
    return d

def cosineAndCorrelation():
    v=np.dot(p[0],p[1])
    v0=np.dot(p[0],p[0])
    v1=np.dot(p[1],p[1])
    cosine=v/math.sqrt(v0*v1)
    #相关系数和标准化后的1,2属性的角度余弦值相等
    plt.scatter(x[0], x[1],s=0.1)
    plt.xlabel("第一属性")
    plt.ylabel("第二属性")
    plt.show()
    return cosine

def normfun(x0, mu, sigma):
    # normfun正态分布函数，mu: 均值，sigma:标准差，pdf:概率密度函数，np.exp():概率密度函数公式
    pdf = np.exp(-((x0 - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

def plotNormalDis():
    # 计算属性1的均值与标准差
    mean=np.mean(x[0])
    std=np.std(x[0])
    # x的范围为，以0.01为单位,需x根据范围调试
    a0 = np.arange(0, 335, 0.01)
    b = normfun(a0, mean, std)
    plt.plot(a0,b, color='g',linewidth = 1)
    #数据，数组，颜色，颜色深浅，组宽，显示频率
    plt.hist(x[0], bins=300,color='r',alpha=0.5,rwidth= 0.9,density=True)
    plt.xlabel("第一属性值")
    plt.ylabel("概率")
    plt.show()

def variance():
    vari=[0 for t in range(10)]
    for i in range(10):
        vari[i]=np.var(a[i])
    print("方差最大的属性是：",argmax(vari)+1,"，方差最小的属性是：",argmin(vari)+1)

def covariance():
    li=[]
    for m in range (10):
        for n in range (m+1,10):
            d=np.dot(p[m],p[n])/19019
            li.append(d)
    print("协方差最大的两组属性是：0和",argmax(li)+1,"；协方差最小的两组属性是：0和",argmin(li)+1)

def main():
    print('数据均值向量：\n',z)
    print('内积算协方差矩阵：\n',inner())
    print('外积算协方差矩阵：\n',outer())
    co=cosineAndCorrelation()
    print('属性1、2的相关性为：',co)
    plotNormalDis()
    variance()
    covariance()

if __name__ == '__main__':
    main()