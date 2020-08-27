#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


print(os.listdir())


# In[3]:


#import our libraries
import numpy as np
import pandas as pd


# In[4]:


#import dataset
data = pd.read_csv('groupStudy.csv')


# In[5]:


data


# In[6]:


X = data.iloc[: , 1:2 ].values
y = data.iloc[:, 2].values


# In[9]:


X


# In[10]:


#Fit data into Decision Tree regressor
from sklearn.tree import DecisionTreeRegressor


# In[11]:


regressor = DecisionTreeRegressor()
regressor.fit(X, y)


# In[16]:


y_predicted = regressor.predict(20)


# In[17]:


y_predicted


# In[ ]:




