import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# %matplotlib inline

def readdata(filename):
    df=pd.read_csv(filename,sep='\t')
    df.drop('Symbol', axis=1, inplace=True)
    S=df.values
    return S

def logReturns(B,random_values,S):
    log_return_conc = np.empty(shape=[0, 100])
    for i in range(0,B):
        random_start = np.random.choice(random_values,replace=True)
        block=S[:,random_start:random_start+30]
        S_first=np.reshape(block[:,[0]],(1,100))
        S_last=np.reshape(block[:,[-1]],(1,100))
        log_return=np.log(S_first)-np.log(S_last)
        log_return_conc=np.vstack([log_return_conc,log_return])
        log_return_conc=np.matrix(log_return_conc)
    return log_return_conc

def top5assets(M_Total):
    meanValue=np.mean(M_Total,axis=0)
    top5assetNumbers=np.argsort(meanValue)[:,-5:]
    top5assetsSymbols=[]
    for x in np.nditer(top5assetNumbers[0]):
        top5assetsSymbols.append("SYM"+str(x))
    return top5assetsSymbols,top5assetNumbers

def boxPlot(log_return_conc,top5assetNumbers,top5assetsSymbols):
    names={}
    print(top5assetNumbers)
    for i,j in  zip(top5assetsSymbols,np.nditer(top5assetNumbers[0])):
        names[i]=log_return_conc[:,[j]].reshape(1,16).tolist()[0]
    plt.boxplot(names.values(),labels=top5assetsSymbols)
    plt.show()
    
def nonParameterMovingBlock(filename):
    S=readdata(filename)
    B=16
    np.random.seed(5000)
    M_Total=np.empty(shape=[0, 100])
    for p in range(0,1000): 
        random_values=np.random.randint(0,1500 , size=1500)
        log_return_conc =logReturns(B,random_values,S)
        column_sum=np.sum(log_return_conc,axis=0)
        M_REV=np.subtract(np.exp(column_sum),1)
        M_Total=np.vstack([M_Total,M_REV])
    top5assetsSymbols,top5assetNumbers=top5assets(M_Total)
    # print(top5assetsSymbols)
    boxPlot(log_return_conc,top5assetNumbers,top5assetsSymbols)
    
nonParameterMovingBlock('Input_Data_Non_Param_BootStrap.txt')