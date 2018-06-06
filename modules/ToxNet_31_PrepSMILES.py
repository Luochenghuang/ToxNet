
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from rdkit import Chem
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


homedir = os.path.dirname(os.path.realpath('__file__'))
homedir = homedir+"/data/"


# In[4]:


jobname = "tox_niehs"
smiles_cutoff = 250


# In[5]:


for taskname in ["verytoxic", "nontoxic", "epa", "ghs", "logld50"]:
    
    filename=homedir+jobname+"_int_"+taskname+".csv"
    fileout=homedir+jobname+"_int_"+taskname+"_smiles.csv"
    data = pd.read_csv(filename)
    X_cut = data[data['smiles'].map(len) < smiles_cutoff]
    print("Database reduced to SMILES length under 250 from "+str(data.shape)+" to "+str(X_cut.shape))
    X = X_cut[["smiles",taskname]]
    X.to_csv(fileout, index=False)


# In[6]:


for taskname in ["verytoxic", "nontoxic", "epa", "ghs", "logld50"]:
    
    filename=homedir+jobname+"_tv_"+taskname+".csv"
    fileout=homedir+jobname+"_tv_"+taskname+"_smiles.csv"
    data = pd.read_csv(filename)
    X_cut = data[data['smiles'].map(len) < smiles_cutoff]
    print("Database reduced to SMILES length under 250 from "+str(data.shape)+" to "+str(X_cut.shape))
    X = X_cut[["smiles",taskname]]
    X.to_csv(fileout, index=False)

