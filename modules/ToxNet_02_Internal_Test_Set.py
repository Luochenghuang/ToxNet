
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from rdkit import Chem
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


homedir = os.path.dirname(os.path.realpath('__file__'))
homedir = homedir+"/data/"
df = pd.read_csv(homedir+"tox_niehs_all.csv")


# In[17]:


df.head()


# # Construct Internal Test Set

# In[18]:


size = 0.10
seed = 6
np.random.seed(seed)


# In[19]:


msk = np.random.rand(len(df)) < 0.1
df_tv = df[~msk]
df_int = df[msk]


# In[20]:


print(df.shape, df_tv.shape, df_int.shape)


# In[21]:


df_tv.to_csv(homedir+'tox_niehs_all_trainval.csv', index=False)
df_int.to_csv(homedir+'tox_niehs_all_int.csv', index=False)


# # Evaluate Dataset Characteristics

# In[22]:


import matplotlib.pyplot as plt


# In[23]:


task = 'verytoxic'

fig, axes = plt.subplots(nrows=1, ncols=3)

df[task].hist(normed=True, ax=axes[0])
df_tv[task].hist(normed=True, ax=axes[1])
df_int[task].hist(normed=True, ax=axes[2])


# In[24]:


task = 'nontoxic'

fig, axes = plt.subplots(nrows=1, ncols=3)

df[task].hist(normed=True, ax=axes[0])
df_tv[task].hist(normed=True, ax=axes[1])
df_int[task].hist(normed=True, ax=axes[2])


# In[25]:


task = 'epa'

fig, axes = plt.subplots(nrows=1, ncols=3)

df[task].hist(normed=True, ax=axes[0])
df_tv[task].hist(normed=True, ax=axes[1])
df_int[task].hist(normed=True, ax=axes[2])


# In[26]:


task = 'ghs'

fig, axes = plt.subplots(nrows=1, ncols=3)

df[task].hist(normed=True, ax=axes[0])
df_tv[task].hist(normed=True, ax=axes[1])
df_int[task].hist(normed=True, ax=axes[2])


# In[27]:


task = 'ld50'

fig, axes = plt.subplots(nrows=1, ncols=3)

df[task].hist(normed=True, ax=axes[0])
df_tv[task].hist(normed=True, ax=axes[1])
df_int[task].hist(normed=True, ax=axes[2])


# In[28]:


task = 'logld50'

fig, axes = plt.subplots(nrows=1, ncols=3)

df[task].hist(normed=True, ax=axes[0])
df_tv[task].hist(normed=True, ax=axes[1])
df_int[task].hist(normed=True, ax=axes[2])

