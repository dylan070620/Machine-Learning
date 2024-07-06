import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path =  'C:/Users/Dylan/Desktop/Machine_Learning/linear_regression/ex1data1.txt'#读取文件用/斜杠
data = pd.read_csv(path, header=None, names=['Population', 'Profit']) #读取csv文件 header没有表头 names【列1，列2】     

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
#plt.show()

def computeCost(X,y,theta):#计算cost func
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))

data.insert(0,'Ones',1)

# set X (training data) and y (target variable)
cols = data.shape[1] #0是矩阵行数 1是列数
X = data.iloc[:,0:cols-1]#X是所有行，去掉最后一列 population那列 输入值
y = data.iloc[:,cols-1:cols]#X是所有行，最后一列 profit那列 真实值
#初始化矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
print(X,y)
#单变量线性回归theta 是一个(1,2)矩阵 w1和b
print(theta.shape)

print(computeCost(X,y,theta))#计算cost 


#batch gradient decent批量梯度下降

def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])#把theta矩阵拉成一列
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X*theta.T)-y

        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

alpha = 0.01#学习率
iters = 1000#学习次数
g,cost = gradientDescent(X, y, theta, alpha, iters)


#绘制拟合
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()