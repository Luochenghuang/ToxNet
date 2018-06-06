
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from rdkit import Chem
get_ipython().run_line_magic('matplotlib', 'inline')


# In[60]:


homedir = os.path.dirname(os.path.realpath('__file__'))+"\\data\\"
ref_df = pd.read_csv(r'{}'.format(homedir+"tox_niehs_desc_minmax.csv"))


# In[61]:


ref_df.head(5)


# # Generate MLP Dataset

# In[64]:


# Find computed descriptors and generate dataset for MLP input data

def create_array(filelist, task):

    for name in filelist:
        df = pd.read_csv(homedir+name+".csv")
        df = df.drop('smiles', axis=1)
        combined_df = pd.merge(df, ref_df, how="left", on=["id"])
        combined_df = combined_df.drop('id', axis=1)
        combined_df.to_csv(homedir+name+"_rdkit.csv", index=False)


# In[63]:


filelist = ['tox_niehs_tv_verytoxic',
            'tox_niehs_int_verytoxic']

create_array(filelist, "verytoxic")


# In[65]:


filelist = ['tox_niehs_tv_nontoxic',
            'tox_niehs_int_nontoxic']

create_array(filelist, "nontoxic")


# In[66]:


filelist = ['tox_niehs_tv_epa',
            'tox_niehs_int_epa']

create_array(filelist, "epa")


# In[67]:


filelist = ['tox_niehs_tv_ghs',
            'tox_niehs_int_ghs']

create_array(filelist, "ghs")


# In[68]:


filelist = ['tox_niehs_tv_logld50',
            'tox_niehs_int_logld50']

create_array(filelist, "logld50")

