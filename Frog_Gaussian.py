
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# In[8]:


dataset=pd.read_csv("Frogs_MFCCs.csv")


# In[9]:


#Retrieving columns 2 to 22
dataset=dataset.iloc[:,1:22]


# In[10]:


g_sum=[]
print ("Total sum within/Totalsum ratios for clusters from 1 to 10 - Gaussian Mixture Models")
for k in range(1,11):
    n_c=k
    G_model = GaussianMixture(n_components = n_c,reg_covar = 1e-2) 
    G_model.fit(dataset)
    

    a=dataset.mean()
    TSS=pd.DataFrame()
    for p in  range(len(dataset.iloc[0,:])):
        TSS[p]=(dataset.iloc[:,p]-a[p])**2   
    c=TSS.sum()
    Totalsum=pd.Series(c).sum()
    G_model.fit(dataset)
    y=G_model.predict(dataset)
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
    g_sum.append(TTSSW/Totalsum)
n=[1,2,3,4,5,6,7,8,9,10]
plt.plot(n,g_sum)
plt.show()

