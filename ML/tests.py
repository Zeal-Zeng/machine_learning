import numpy as np
import pylab as pl



data = np.loadtxt('.\data\\test_liner_data.txt', delimiter=',')
x = data[:, 0]
y = data[:, 1]
x.shape = (len(x),1)
y.shape = (len(y),1)
pl.scatter(x, y)
b = np.ones((len(x), 1))
x = np.hstack((x, b))

m = len(x)
diff = [0,0]
#learning rate
alpha = 0.01

#init the parameters to zero
theta0 = 0
theta1 = 0
sum0=0
sum1=0

theta = np.zeros((2,1))
for h in range(1000):


    # for i in range(m):
    #     diff[0] = y[i]-( theta0 * x[i][0] + theta1* x[i][1]   )
    #     sum0+= alpha * diff[0] * x[i][0]
    #     sum1+= alpha * diff[0]* x[i][1]
    # theta0 += sum0
    # theta1+= sum1
    for i in range(m):
        diff[0] = y[i]-( theta0 * x[i][0] + theta1* x[i][1]   )
        theta0 = theta0 + alpha * diff[0] * x[i][0]
        theta1 = theta1 + alpha * diff[0]* x[i][1]
i = pl.linspace(0,10)
j = theta0*i+theta1
pl.plot(i,j)
pl.show()