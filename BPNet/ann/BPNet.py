from math import e
import numpy as np

def sigmoid(inX):
    return 1.0 / (1 + e**(-inX))

def dsigmoid(inX):
    return (1-sigmoid(inX))*sigmoid(inX)

class BPNet():
    def __init__(self,size,isNeedBias):
        '''
        :param size: list (输入数 隐层数 输出数)
        :return:
        '''
        self.weight = []
        self.isNeedBias = isNeedBias
        pre_layer_size = size[0]
        for current_layer_size in size[1:]:
            if isNeedBias:
                self.weight.append(2*np.random.random((pre_layer_size+1,current_layer_size))- 1)
            else:
                self.weight.append(2*np.random.random((pre_layer_size,current_layer_size))- 1)
            pre_layer_size = current_layer_size


    def run(self,X):
        '''
        :param X: 每一行代表一个向量
        :return: 对Y的预测
        '''
        temp_x = X
        for w in self.weight:
            if self.isNeedBias:
                temp_x = np.hstack((temp_x,np.ones(( len(temp_x),1 ))))  #加上偏执
            temp_x = temp_x.dot(w)#得到和函数
            temp_x = sigmoid(temp_x) #激活输出
        return temp_x

    def training(self,X,Y):
        for x,y in zip(X,Y):
            x.shape = (1, len(x) )
            y.shape = (1, len(y) )
            d_sum = [] #更新权值时所需的激活函数的导
            Xs = [] #每层的输入
            temp_x = x #每一层输入的临时变量

            #正向---------------------------------->
            for w in self.weight:
                if self.isNeedBias:
                    temp_x = np.hstack((temp_x ,np.ones(( len([temp_x]),1 ))))  #加上偏执
                Xs.append(temp_x)
                temp_x = temp_x.dot(w) #得到和函数
                d_sum.append(dsigmoid(temp_x)) #计算和函数的导
                temp_x = sigmoid(temp_x) #计算输出

            #反向----------------------------------->
            # Xs.append(temp_x)
            error = temp_x - y #计算误差
            delta = []
            delta.append(error * d_sum[len(d_sum)-1])#计算最后一层的delta
            for i in range(len(self.weight)-1):
                w = self.weight[ len(self.weight) -1-i ]
                sigma = delta[i].dot(w.T) #残差加权和
                if self.isNeedBias:
                    delta.append(sigma[:,0:len(sigma[0])-1]* d_sum[len(d_sum)-2-i] )
                else:
                    delta.append(sigma* d_sum[len(d_sum)-2-i] )

            #更新权值
            for i in range(len(Xs)):
                xx = Xs[i]
                self.weight[i] -= 0.28*xx.T.dot(delta[len(delta) -1-i])
            # print(self.weight[i] )

    def calculate_error(self,X ,Y):
        sum_error = 0.0
        for h,y in zip(self.run(X),Y):
            E=0.0
            for e in h-y:
                E += e
            sum_error += E*0.1
        return sum_error/len(X)







