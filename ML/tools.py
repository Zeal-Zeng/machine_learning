from math import e

import numpy as np


def sigmoid(X):
    return 1.0/(1 + e ** ( (-1.0)*X) )

def gradient_descent_1(theta, X , y):
    ''' J() =   sigma(( (h() - y) ** 2 )) /(2* m)
        J'() = sigma( (h() - y)*x  ) /m
    '''
    cost = X.dot(theta.T) - y
    cost = (cost).T.dot(X)
    # s.dot(X.T)
    return (1.0/len(X))*cost


def gradient_descent(x, y, theta):
    theta.shape=(len(theta),1)
    newtheta = theta
    for i, j in zip(x,y):
        dif = sigmoid(i.dot(newtheta))-j
        i.shape = (len(i),1)
        newtheta -= 1*dif*i
    # print(newtheta)
    return newtheta

def gradient_descent_2(theta, X , y):
    '''
        当y取1，0 时
        h() = sigmoid(θ.T * x)
        J(θ) =  -( sigma(y* ln(h(x)) + (1-y)ln( 1 - h(x) ) ) )/m
        J'(θ) = (1.0/m) sigma( h(x) - y )*x
    '''

    h = sigmoid( X.dot(theta.T) ) #算h(x)
    delta = (h-y)   #差值

    sigma = X.T.dot(delta)  #与Xi相乘求和
    # print(delta.T.shape,X.shape)
    return (1.0/len(X))*sigma

def compute_grad(theta, X, y):
    '''''compute gradient'''
    theta.shape =(1,3)
    grad = np.zeros(3)
    h = sigmoid(X.dot(theta.T))
    h.shape = (len(h),1)
    y.shape = (len(y),1)
    delta = h - y
    l = grad.size
    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i]=(1.0/ len(X))* sumdelta *-1
    theta.shape =(3,)
    return  grad