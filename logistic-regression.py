#encoding:utf-8

#Logistic回归梯度上升最优化算法
import math
from numpy import *



def loadDataSet():
	dataMat = []; labelMat = []
	fr = open(data_file_path)
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat

def sigmoid(inX):
	return 1.0/(1+exp(-inX))

#批量梯度上升算法(batch gradient ascend)
def gradAscent(dataMatIn,classLabels):
	#调用numpy函数可以将数组转行为矩阵数据类型
	dataMatrix = mat(dataMatIn)
	#将矩阵转置
	labelMat = mat(classLabels).transpose()
	#看一下dataMatrix的大小，3*100的矩阵
	m,n = shape(dataMatrix)
	#目标移动步长
	alpha = 0.01
	#迭代次数
	maxCycles =500
	#n已经赋值为3
	weights = ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)
		error = (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights

# 随机梯度上升算法(stochastic gradient ascend)
def stocGradAscent0(dataMatrix,classLabels):
	m,n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i]*weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights

#前者(BGD)的h和误差error都是向量，而后者(SGD)则是数值


"""
改进的随机梯度算法
alpha在每次迭代的时候都会调整，这会缓解数据波动或者高频波动。
通过随机选取样本来更新回归系数，这样可以减少周期性波动
增加了一个迭代参数作为第三个参数，算法将按照给的新的参数值进行迭代
"""
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
   m,n = shape(dataMatrix)
   weights = ones(n)
"""
   i 是样本点的下标,j 是迭代次数
   """
   for j in range(numIter):         
       dataIndex = range(m)
       for i in range(m):
           alpha = 4/(1.0 + j + i) + 0.01  #alpha每次迭代时需要调整
           randIndex = int(random.uniform(0,len(dataIndex)))
           h = sigmoid(sum(dataMatrix[randIndex]*weights))
           error = classLabels[randIndex] - h
           weights = weights + alpha *error * dataMatrix[randIndex]
           del(dataIndex[randIndex])
   return weights