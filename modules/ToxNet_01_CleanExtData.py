
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from rdkit import Chem
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


homedir = os.path.dirname(os.path.realpath('__file__'))
homedir = homedir+"/data/"
df = pd.read_csv(homedir+"validationset.txt", sep='\t')


# In[18]:


# Add unique alphanumeric identifier
df['id'] = range(1, len(df.index)+1)
df['id'] = 'testid' + df['id'].astype(str)
print(df.shape)


# In[19]:


df.head(5)


# # Ensure all SMILES are legit, and canonicalize SMILES

# In[20]:


# Remove extraneous SMILES entry
df = df.join(df['Canonical_QSARr'].str.split(' ', 1, expand=True).rename(columns={0:'pre_smiles', 1:'Extraneous_SMILES'}))
df.head(5)


# In[21]:


# Check for invalid SMILES
mol_list = [Chem.MolFromSmiles(x) for x in df['pre_smiles']]
invalid = len([x for x in mol_list if x is None])
print("No. of invalid entries: "+str(invalid))


# In[22]:


# Canonicalize SMILES
newdf = []
for index, row in df.iterrows():
    smiles_string = df['pre_smiles'][index]
    mol = Chem.MolFromSmiles(smiles_string)
    newdf.append(Chem.MolToSmiles(mol))


# In[23]:


# Replace SMILES with canonicalized versions
add_df = pd.DataFrame(np.asarray(newdf),columns=["smiles"])
print(df.shape)
df = pd.concat([df, add_df], axis=1)
print(df.shape)
df = df.drop(['pre_smiles'], axis=1)
print(df.shape)


# # Standardize labels

# In[24]:


# Rename columns
df = df.rename(columns={'very_toxic': 'verytoxic','nontoxic': 'nontoxic',                    'EPA_category': 'epa','GHS_category': 'ghs','LD50_mgkg': 'ld50'})
df.head(5)


# In[25]:


# Replace T/F with integers
df['verytoxic'].replace(False, 0, inplace=True)
df['verytoxic'].replace(True, 1, inplace=True)
df['nontoxic'].replace(False, 0, inplace=True)
df['nontoxic'].replace(True, 1, inplace=True)


# In[26]:


# Rename EPA/GHS category to start from zero
df['epa'] = df['epa'] - 1
df['ghs'] = df['ghs'] - 1


# In[27]:


# Apply log transformation to ld50
df['logld50'] = np.log(df['ld50'])


# In[28]:


df.to_csv(homedir+"tox_niehs_ext_raw.csv", index=False)
df.head(5)

