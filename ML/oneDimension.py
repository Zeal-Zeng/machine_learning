import numpy as np
import pylab as pl


data = np.loadtxt('.\data\data1D.txt', delimiter=',')
x = data[:, 0]
y = data[:, 1]
x.shape = (len(x),1)
y.shape = (len(y),1)
pl.scatter(x, y)
b = np.ones((len(x), 1))

x = np.hstack((x, b))

#采用 y = theta[0] * X +theta[1]
theta = np.zeros((2,1))
# d = np.ones((len(x),1))*5 #2.7
# x = (x - d)/2.7

def gradient_descent(x, y, theta):
    newtheta = theta
    for i, j in zip(x,y):
        dif = i.dot(newtheta)-j
        i.shape = (len(i),1)
        newtheta -= 0.06*dif*i
    print(newtheta)
    return newtheta

        # print(dif)
        # print(i,j)
    # for i in range(len(x)):
    #     x[i].shape = (1,2)
    #     dif = x[i].dot(newtheta)
    #     print(dif.shape)
    #     print(y[i].shape)
    #     newtheta -= (dif-y[i])*x[i]
    # return newtheta
    # return (1.0/len(x))*x.T.dot(x.dot(theta)-y)

for i in range(40):
    theta = gradient_descent(x,y,theta)

i = pl.linspace(0,10)
j = theta[0]*i+theta[1]

pl.plot(i,j)
pl.show()