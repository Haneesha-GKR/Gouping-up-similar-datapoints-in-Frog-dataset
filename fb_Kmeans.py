
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[33]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[34]:


dataset=pd.read_csv("dataset_Facebook.csv",sep=';')


# In[35]:


le.fit(["Photo","Status","Link","Video"])
dataset['Type']=le.transform(dataset['Type'])


# In[36]:



dataset= dataset.dropna()


# In[37]:


K_sum=[]
print ("Total sum within/Totalsum ratios for clusters from 1 to 10 - KMeans Clustering")
for k in range(1,11):
    n_c=k
    k_model = KMeans(n_c) 
    k_model.fit(dataset)
    
    
    x=dataset.mean()
    TSS=pd.DataFrame()
    for p in  range(len(dataset.iloc[0,:])):
        TSS[p]=(dataset.iloc[:,p]-x[p])**2   
    c=TSS.sum()
    Totalsum=pd.Series(c).sum()
    y=k_model.fit_predict(dataset)
    dataset['cluster_number']=y
    grp=dataset.groupby(["cluster_number"]).mean()
    SSW=[]
    for p in range(n_c):
        B=pd.DataFrame()
        B=dataset.loc[dataset['cluster_number'] == p]
        b=B.mean()
        TSSW=pd.DataFrame()
        for p in  range(len(B.iloc[0,:])):
            TSSW[p]=(B.iloc[:,p]-b[p])**2   
        c=TSSW.sum()
        SSW.append(pd.Series(c).sum())
    TTSSW=sum(SSW)
    print(TTSSW/Totalsum)
    K_sum.append(TTSSW/Totalsum)
n=[1,2,3,4,5,6,7,8,9,10]
plt.plot(n,K_sum)
print()

