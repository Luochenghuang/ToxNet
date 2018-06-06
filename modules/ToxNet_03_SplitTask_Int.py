
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from rdkit import Chem
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


homedir = os.path.dirname(os.path.realpath('__file__'))
homedir = homedir+"/data/"
df = pd.read_csv(homedir+"tox_niehs_all_int.csv")


# # Split data by task

# In[5]:


# There are 5 measurements: very_toxic, nontoxic, EPA_category, GHS_category and LD50_mgkg
# Make dataset for each of the measurements separately


# In[6]:


# Very Toxic


# In[7]:


# Check for missing labels
df['verytoxic'].isnull().sum()


# In[8]:


# Drop data with missing labels
df1 = df[['id','smiles','verytoxic']]
df1 = df1.dropna(axis=0, how='any')
print("Missing data:" +str(df1['verytoxic'].isnull().sum()))
df1.to_csv(homedir+"tox_niehs_int_verytoxic.csv", index=False)
df1.groupby('verytoxic').count()


# In[9]:


# Non Toxic


# In[10]:


# Check for missing labels
df['nontoxic'].isnull().sum()


# In[11]:


# Drop data with missing labels
df1 = df[['id','smiles','nontoxic']]
df1 = df1.dropna(axis=0, how='any')
print("Missing data:" +str(df1['nontoxic'].isnull().sum()))
df1.to_csv(homedir+"tox_niehs_int_nontoxic.csv", index=False)
df1.groupby('nontoxic').count()


# In[12]:


# EPA


# In[13]:


df['epa'].isnull().sum()


# In[14]:


# Drop data with missing labels
df1 = df[['id','smiles','epa']]
df1 = df1.dropna(axis=0, how='any')
print("Missing data:" +str(df1['epa'].isnull().sum()))
df1.to_csv(homedir+"tox_niehs_int_epa.csv", index=False)
df1.groupby('epa').count()


# In[15]:


# GHS


# In[16]:


df['ghs'].isnull().sum()


# In[17]:


# Drop data with missing labels
df1 = df[['id','smiles','ghs']]
df1 = df1.dropna(axis=0, how='any')
print("Missing data:" +str(df1['ghs'].isnull().sum()))
df1.to_csv(homedir+"tox_niehs_int_ghs.csv", index=False)
df1.groupby('ghs').count()


# In[18]:


# LD50


# In[19]:


df['logld50'].isnull().sum()


# In[20]:


# Drop data with missing labels
df1 = df[['id','smiles','logld50']]
df1 = df1.dropna(axis=0, how='any')
print("Missing data:" +str(df1['logld50'].isnull().sum()))
df1.to_csv(homedir+"tox_niehs_int_logld50.csv", index=False)
print(df1['logld50'].max(axis=0), df1['logld50'].min(axis=0))

