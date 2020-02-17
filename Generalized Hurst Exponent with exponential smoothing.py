import pandas as pd
import numpy as np
import math
def readInputData(filename):
    df = pd.read_csv(filename, header=None)
    S=df.values.reshape(1,1000)
    return S

def getKvalue(S,theta):
    alpha=1/float(theta)
    k=np.zeros((1,19))
    for t_max in range(1,20):
        t=1000-t_max
        w_numer=1-math.exp(-alpha)
        w_denom=1-math.exp(-alpha*t)
        w0=w_numer/w_denom
        w1=w0*np.exp(np.arange(0,t-1)/theta)
        n_t=abs(S[0][np.arange(0,t-1)+t_max] - S[0][np.arange(0,t-1)])
        d_t=abs(S[0][np.arange(0,t-1)])
        n_tq=np.square(n_t)
        d_tq=np.square(d_t)
        n_avg=np.sum(np.multiply(w1[::-1],n_tq))
        d_avg=np.sum(np.multiply(w1[::-1],d_tq))
        k[0][t_max-1]=n_avg/d_avg
    return k[0]

def linearReg(K,T_max):
    b=np.zeros((len(range(5,20)),1))
    indx=0
    for j in range(5,20):
        t_j=T_max[np.arange(j)]
        k_j=K[np.arange(j)]
        y=np.log(k_j)
        x=np.log(t_j)
        b_xy= np.sum(np.multiply(x,y)) - j*np.mean(x)*np.mean(y)  
        b_xx= np.sum(np.multiply(x,x)) - j*np.mean(x)*np.mean(x)
        b[indx]=b_xy/b_xx
        indx+=1
    return b

def generalizedHurstExponent(filename):
    theta=100
    q=2
    T_max=np.arange(1,20)
    S=readInputData(filename)
    K=getKvalue(S,theta)
    b=linearReg(K,T_max)
    b_avg=np.mean(b)
    return b_avg/q

if __name__ == "__main__":
    inputfilenames=['GHE_1.txt','GHE_2.txt','GHE_3.txt','GHE_4.txt']
    for i in inputfilenames:
        value=generalizedHurstExponent(i)
        print(value)