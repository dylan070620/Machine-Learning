import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path =  'C:/Users/Dylan/Desktop/Machine_Learning/linear_regression/ex1data2.txt'#读取文件用/斜杠
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
#特征归一化
data2 = (data2 - data2.mean()) / data2.std()
# add ones column
data2.insert(0, 'Ones', 1)
# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]
# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

def computeCost(X,y,theta):#计算cost func
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))
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
# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
computeCost(X2, y2, g2)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()