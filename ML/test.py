import numpy as np
import pylab as pl
from tools import gradient_descent_1, gradient_descent_2, sigmoid ,compute_grad,gradient_descent

data = np.loadtxt('.\data\ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
pos = np.where(y == 1)  #返回一个tuple y==1时 Y的位置
neg = np.where(y == 0)
pl.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
pl.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
pl.xlabel('Feature1/Exam 1 score')
pl.ylabel('Feature2/Exam 2 score')
pl.legend(['Fail', 'Pass'])

b = np.ones((len(X), 1))*60
X = np.hstack((X,b))

# theta = np.array([2,0,1.5])
theta = np.zeros(3)
for m in range(1222):
    theta = gradient_descent(X, y, theta )
    # print(theta)
i = np.linspace(30, 90)
j = -(theta[0]*i+theta[2])/theta[1]
pl.plot(i,j)
print('\n')
print(theta[0],theta[1],theta[2])
theta = np.array([-1,-1,120])


def predict(theta, X):
    '''''Predict label using learned logistic regression parameters'''
    # m, n = X.shape
    p = np.zeros(len(X))
    # print(X.dot(theta.T))
    h = sigmoid(X.dot(theta.T))
    # print(h)
    for it in range(len(h)):
        if h[it]>0.5:
            p[it]=1
        else:
            p[it]=0
    return p
#Compute accuracy on our training set
p = predict(theta, X)
print('\n')




sum =0
for i,j in zip(p,y):
    if(i==j):
        sum+=1

print(sum, len(X))

print()
i = np.linspace(0, 120)
j = -(theta[0]*i+theta[2])/theta[1]
# j = -i + 120
# pl.plot(i,j)
pl.show()