
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from rdkit import Chem
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


homedir = os.path.dirname(os.path.realpath('__file__'))
homedir = homedir+"/data/"
df = pd.read_csv(homedir+"trainingset_171130.txt", sep='\t')


# In[10]:


# Add unique alphanumeric identifier
df['id'] = range(1, len(df.index)+1)
df['id'] = 'molid' + df['id'].astype(str)
print(df.shape)


# In[11]:


df.head(5)


# # Ensure all SMILES are legit, and canonicalize SMILES

# In[12]:


# Remove extraneous SMILES entry
df = df.join(df['Canonical_QSARr'].str.split(' ', 1, expand=True).rename(columns={0:'pre_smiles', 1:'Extraneous_SMILES'}))
df.head(5)


# In[13]:


# Check for invalid SMILES
mol_list = [Chem.MolFromSmiles(x) for x in df['pre_smiles']]
invalid = len([x for x in mol_list if x is None])
print("No. of invalid entries: "+str(invalid))


# In[14]:


# Canonicalize SMILES
newdf = []
for index, row in df.iterrows():
    smiles_string = df['pre_smiles'][index]
    mol = Chem.MolFromSmiles(smiles_string)
    newdf.append(Chem.MolToSmiles(mol))


# In[15]:


# Replace SMILES with canonicalized versions
add_df = pd.DataFrame(np.asarray(newdf),columns=["smiles"])
print(df.shape)
df = pd.concat([df, add_df], axis=1)
print(df.shape)
df = df.drop(['pre_smiles'], axis=1)
print(df.shape)


# # Standardize labels

# In[16]:


# Rename columns
df = df.rename(columns={'very_toxic': 'verytoxic','nontoxic': 'nontoxic',                    'EPA_category': 'epa','GHS_category': 'ghs','LD50_mgkg': 'ld50'})
df.head(5)


# In[17]:


# Replace T/F with integers
df['verytoxic'].replace(False, 0, inplace=True)
df['verytoxic'].replace(True, 1, inplace=True)
df['nontoxic'].replace(False, 0, inplace=True)
df['nontoxic'].replace(True, 1, inplace=True)


# In[18]:


# Rename EPA/GHS category to start from zero
df['epa'] = df['epa'] - 1
df['ghs'] = df['ghs'] - 1


# In[19]:


# Apply log transformation to ld50
df['logld50'] = np.log(df['ld50'])


# In[20]:


df.to_csv(homedir+"tox_niehs_all_raw.csv", index=False)
df.head(5)


# # Deal with duplicate entries

# In[21]:


mask = df.duplicated('smiles', keep=False)


# In[22]:


#Separate out unique and duplicate entries
df_uni = df[~mask]
df_dup = df[mask]
print(df.shape, df_uni.shape, df_dup.shape)


# In[23]:


# Compute mean of duplicate entries
avg_df = df_dup.groupby('smiles', as_index=False).mean()
avg_df.head(25)


# In[24]:


# Drop unreliable labels
print(avg_df.shape)
avg_df = avg_df[avg_df["verytoxic"] != 0.5]
print(avg_df.shape)
avg_df = avg_df[avg_df["nontoxic"] != 0.5]
print(avg_df.shape)
avg_df = avg_df[avg_df["epa"] != 0.5]
avg_df = avg_df[avg_df["epa"] != 1.5]
avg_df = avg_df[avg_df["epa"] != 2.5]
avg_df = avg_df[avg_df["epa"] != 3.5]
print(avg_df.shape)
avg_df = avg_df[avg_df["ghs"] != 0.5]
avg_df = avg_df[avg_df["ghs"] != 1.5]
avg_df = avg_df[avg_df["ghs"] != 2.5]
avg_df = avg_df[avg_df["ghs"] != 3.5]
avg_df = avg_df[avg_df["ghs"] != 4.5]
print(avg_df.shape)
avg_df.head(25)


# In[25]:


# Round to nearest integer (select nearest category)
avg_df = avg_df.round({'verytoxic': 0, 'nontoxic': 0, 'epa': 0, 'ghs':0})
avg_df.head(25)


# In[26]:


# Match up average predictions to SMILES and drop duplicate entries
print(df_dup.shape)
df_dup = df_dup.drop(['verytoxic', 'nontoxic', 'epa', 'ghs', 'ld50', 'logld50'], axis=1)
df_dup = pd.merge(df_dup, avg_df, how="right", on=["smiles"])
print(df_dup.shape)
df_dup = df_dup.drop_duplicates(subset=['smiles'], keep="first")
print(df_dup.shape)


# In[27]:


df_dup.head(5)


# In[28]:


# Add reliable averaged de-duplicated entries back to unique entries
df2 = pd.concat([df_dup, df_uni], axis=0)
print(df2.shape)
print(df2.smiles.unique().shape)
print(df.smiles.unique().shape)


# In[29]:


# Reset index of df
df2 = df2.reset_index(drop=True)
df2.head(5)


# In[30]:


df2.to_csv(homedir+"tox_niehs_all.csv", index=False)

