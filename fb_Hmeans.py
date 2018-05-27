
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering



# In[40]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[41]:


dataset=pd.read_csv("dataset_Facebook.csv",sep=';')


# In[42]:


le.fit(["Photo","Status","Link","Video"])
dataset['Type']=le.transform(dataset['Type'])


# In[43]:



dataset= dataset.dropna()


# In[44]:


H_sum=[]
print ("Totalsumwithin/Totalsum ratios for clusters from 1 to 10 - H-Clustering")
for k in range(1,11):
    n_c=k
    h_model = AgglomerativeClustering(n_clusters=n_c) 
    h_model.fit(dataset)
    
    
    a=dataset.mean()
    TSS=pd.DataFrame()
    for p in  range(len(dataset.iloc[0,:])):
        TSS[p]=(dataset.iloc[:,p]-a[p])**2   
    c=TSS.sum()
    Totalsum=pd.Series(c).sum()
    y=h_model.fit_predict(dataset)
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
    H_sum.append(TTSSW/Totalsum)
n=[1,2,3,4,5,6,7,8,9,10]
plt.plot(n,H_sum)
print()

