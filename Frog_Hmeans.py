
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


# In[15]:


#Reading the data
dataset=pd.read_csv("Frogs_MFCCs.csv")


# In[16]:


#Retrieving columns 2 to 22
dataset=dataset.iloc[:,1:22]


# In[17]:


#H-Clustering
H_sum_val=[]
for p in range(1,11):
    n_c=p
    h_model = AgglomerativeClustering(n_clusters=n_c) 
    h_model.fit(dataset)
    
    df= pd.DataFrame([[4,2],[4,5],[7,8],[7,10]])
    x=dataset.mean()
    tss=pd.DataFrame()
    for q in  range(len(dataset.iloc[0,:])):
        tss[q]=(dataset.iloc[:,q]-x[q])**2   
    c=tss.sum()
    Totalsum=pd.Series(c).sum()
    
    y=h_model.fit_predict(dataset)
    dataset['cluster_number']=y
    grouped=dataset.groupby(["cluster_number"]).mean()
    SSW=[]
    for q in range(n_c):
        B=pd.DataFrame()
        B=dataset.loc[dataset['cluster_number'] == q]
        b=B.mean()
        TSSW=pd.DataFrame()
        for q in  range(len(B.iloc[0,:])):
            TSSW[q]=(B.iloc[:,q]-b[q])**2   
        c=TSSW.sum()
        SSW.append(pd.Series(c).sum())
    TTSSW=sum(SSW)
    H_sum_val.append(TTSSW/Totalsum)




# In[18]:


print ("Total sum within/Totalsum ratios for clusters from 1 to 10",H_sum_val)

