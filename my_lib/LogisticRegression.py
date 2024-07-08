import numpy as np
import pandas as pd

def sigmoid(x):
    z=(1/(1+np.exp(-x)))
    return z

class LogisticRegression:

    def __init__(self):
        self.coefficient=None
        self.bias=None

    def f(self,X,a,b):
        fx=sigmoid(np.dot(a,X)+b)
        return fx

    def cost(self,X,Y,a,b):
        m=len(Y)
        cost=sum([(-Y[i]*np.log(self.f(X[i],a,b)) - (1-Y[i])*np.log(1-self.f(X[i],a,b))) for i in range(m)])/(2*m)
        return cost
    
    def grad_desc(self,X,Y,a,b,learning_rate,m,n):
        temp_a=np.zeros(n)
        temp_b=0
        for j in range (n):
            temp_a[j]=a[j]-learning_rate*(1/m)*sum([(self.f(X[i],a,b)-Y[i])*X[i][j] for i in range(m)])
        temp_b=b-learning_rate*(1/m)*sum([(self.f(X[i],a,b)-Y[i]) for i in range(m)])
        return (temp_a,temp_b)

    def train(self,X,Y):
        X=X.values
        Y=Y.values
        m=len(Y)
        n=len(X[0])
        a=np.zeros(n)
        b=0
        learning_rate=0.01
        cost_hist=[self.cost(X,Y,a,b)]
        itr=1000
        for i in range(itr):
            (a,b)=self.grad_desc(X,Y,a,b,learning_rate,m,n)
            cost_hist.append(self.cost(X,Y,a,b))

            # if (i+1)%100==0:
            #     print(f'Iteration={i+1}, Cost={cost_hist[i+1]}, weights={a}, bias={b}')

            epsilon=(abs(cost_hist[i+1]-cost_hist[i])/cost_hist[i+1])*100

            # if epsilon<0.001:
            #     print(f'Iteration={i+1}, Cost={cost_hist[i+1]}, a={a}, b={b}')
            #     self.coefficient=a
            #     self.bias=b
            #     return

        self.coefficient=a
        self.bias=b
        return

    def predict(self,X):
        try:
            X=X.values
        except:
            None
        threshold=0.5
        yhat=sigmoid((self.coefficient@X.T)+self.bias)
        y_predicted = [1 if i > threshold else 0 for i in yhat]
        return y_predicted