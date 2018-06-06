
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


homedir = os.path.dirname(os.path.realpath('__file__'))+"/data/"
df1 = pd.read_csv(homedir+"tox_niehs_all.csv")
df2 = pd.read_csv(homedir+"tox_niehs_ext_raw.csv")


# # Functions for descriptor calculation

# In[4]:


from rdkit.Chem import Descriptors

def compute_descriptors(mol, id_string):

    descriptors = [id_string]
    
    # Property descriptor
    descriptors.append(Descriptors.MolWt(mol))
    descriptors.append(Descriptors.HeavyAtomMolWt(mol))
    descriptors.append(Descriptors.MolLogP(mol))
    descriptors.append(Descriptors.MolMR(mol))
    descriptors.append(Descriptors.TPSA(mol))    
    # Constitutional descriptor
    descriptors.append(Descriptors.FractionCSP3(mol))
    # Atom
    descriptors.append(Descriptors.HeavyAtomCount(mol))
    descriptors.append(Descriptors.NHOHCount(mol))
    descriptors.append(Descriptors.NOCount(mol))
    descriptors.append(Descriptors.NumHAcceptors(mol))
    descriptors.append(Descriptors.NumHDonors(mol))    
    descriptors.append(Descriptors.NumHeteroatoms(mol))
    #descriptors.append(Descriptors.NumBridgeheadAtoms(mol))
    #descriptors.append(Descriptors.NumSpiroAtoms(mol))
    # Bond
    descriptors.append(Descriptors.NumRotatableBonds(mol))
    # Electronic
    descriptors.append(Descriptors.NumRadicalElectrons(mol))
    descriptors.append(Descriptors.NumValenceElectrons(mol))
    descriptors.append(Descriptors.MaxPartialCharge(mol))
    descriptors.append(Descriptors.MinPartialCharge(mol))
    descriptors.append(Descriptors.MaxAbsPartialCharge(mol))
    descriptors.append(Descriptors.MinAbsPartialCharge(mol))
    # Ring
    #descriptors.append(Descriptors.NumRings(mol))
    descriptors.append(Descriptors.NumAromaticRings(mol))
    descriptors.append(Descriptors.NumSaturatedRings(mol))    
    descriptors.append(Descriptors.NumAliphaticRings(mol))
    #descriptors.append(Descriptors.NumCarbocycles(mol))
    descriptors.append(Descriptors.NumAromaticCarbocycles(mol))
    descriptors.append(Descriptors.NumSaturatedCarbocycles(mol))  
    descriptors.append(Descriptors.NumAliphaticCarbocycles(mol))  
    #descriptors.append(Descriptors.NumHeterocycles(mol))
    descriptors.append(Descriptors.NumAromaticHeterocycles(mol))
    descriptors.append(Descriptors.NumSaturatedHeterocycles(mol)) 
    descriptors.append(Descriptors.NumAliphaticHeterocycles(mol))    
    # Functional Groups
    descriptors.append(Descriptors.fr_Al_COO(mol))
    descriptors.append(Descriptors.fr_Al_OH(mol))
    descriptors.append(Descriptors.fr_Al_OH_noTert(mol))
    descriptors.append(Descriptors.fr_ArN(mol))
    descriptors.append(Descriptors.fr_Ar_COO(mol))
    descriptors.append(Descriptors.fr_Ar_N(mol))
    descriptors.append(Descriptors.fr_Ar_NH(mol))
    descriptors.append(Descriptors.fr_Ar_OH(mol))
    descriptors.append(Descriptors.fr_COO(mol))
    descriptors.append(Descriptors.fr_COO2(mol))
    descriptors.append(Descriptors.fr_C_O(mol))
    descriptors.append(Descriptors.fr_C_O_noCOO(mol))
    descriptors.append(Descriptors.fr_C_S(mol))
    descriptors.append(Descriptors.fr_HOCCN(mol))
    descriptors.append(Descriptors.fr_Imine(mol))
    descriptors.append(Descriptors.fr_NH0(mol))
    descriptors.append(Descriptors.fr_NH1(mol))
    descriptors.append(Descriptors.fr_NH2(mol))
    descriptors.append(Descriptors.fr_N_O(mol))
    descriptors.append(Descriptors.fr_Ndealkylation1(mol))
    descriptors.append(Descriptors.fr_Ndealkylation2(mol))
    descriptors.append(Descriptors.fr_Nhpyrrole(mol))
    descriptors.append(Descriptors.fr_SH(mol))
    descriptors.append(Descriptors.fr_aldehyde(mol))
    descriptors.append(Descriptors.fr_alkyl_carbamate(mol))
    descriptors.append(Descriptors.fr_alkyl_halide(mol))
    descriptors.append(Descriptors.fr_allylic_oxid(mol))
    descriptors.append(Descriptors.fr_amide(mol))
    descriptors.append(Descriptors.fr_amidine(mol))
    descriptors.append(Descriptors.fr_aniline(mol))
    descriptors.append(Descriptors.fr_aryl_methyl(mol))
    descriptors.append(Descriptors.fr_azide(mol))
    descriptors.append(Descriptors.fr_azo(mol))
    descriptors.append(Descriptors.fr_barbitur(mol))
    descriptors.append(Descriptors.fr_benzene(mol))
    descriptors.append(Descriptors.fr_benzodiazepine(mol))
    descriptors.append(Descriptors.fr_bicyclic(mol))
    descriptors.append(Descriptors.fr_diazo(mol))
    descriptors.append(Descriptors.fr_dihydropyridine(mol))
    descriptors.append(Descriptors.fr_epoxide(mol))
    descriptors.append(Descriptors.fr_ester(mol))
    descriptors.append(Descriptors.fr_ether(mol))
    descriptors.append(Descriptors.fr_furan(mol))
    descriptors.append(Descriptors.fr_guanido(mol))
    descriptors.append(Descriptors.fr_halogen(mol))
    descriptors.append(Descriptors.fr_hdrzine(mol))
    descriptors.append(Descriptors.fr_hdrzone(mol))
    descriptors.append(Descriptors.fr_imidazole(mol))
    descriptors.append(Descriptors.fr_imide(mol))
    descriptors.append(Descriptors.fr_isocyan(mol))
    descriptors.append(Descriptors.fr_isothiocyan(mol))
    descriptors.append(Descriptors.fr_ketone(mol))
    descriptors.append(Descriptors.fr_ketone_Topliss(mol))
    descriptors.append(Descriptors.fr_lactam(mol))
    descriptors.append(Descriptors.fr_lactone(mol))
    descriptors.append(Descriptors.fr_methoxy(mol))
    descriptors.append(Descriptors.fr_morpholine(mol))
    descriptors.append(Descriptors.fr_nitrile(mol))
    descriptors.append(Descriptors.fr_nitro(mol))
    descriptors.append(Descriptors.fr_nitro_arom(mol))
    descriptors.append(Descriptors.fr_nitro_arom_nonortho(mol))
    descriptors.append(Descriptors.fr_nitroso(mol))
    descriptors.append(Descriptors.fr_oxazole(mol))
    descriptors.append(Descriptors.fr_oxime(mol))
    descriptors.append(Descriptors.fr_para_hydroxylation(mol))
    descriptors.append(Descriptors.fr_phenol(mol))
    descriptors.append(Descriptors.fr_phenol_noOrthoHbond(mol))
    descriptors.append(Descriptors.fr_phos_acid(mol))
    descriptors.append(Descriptors.fr_phos_ester(mol))
    descriptors.append(Descriptors.fr_piperdine(mol))
    descriptors.append(Descriptors.fr_piperzine(mol))
    descriptors.append(Descriptors.fr_priamide(mol))
    descriptors.append(Descriptors.fr_prisulfonamd(mol))
    descriptors.append(Descriptors.fr_pyridine(mol))
    descriptors.append(Descriptors.fr_quatN(mol))
    descriptors.append(Descriptors.fr_sulfide(mol))
    descriptors.append(Descriptors.fr_sulfonamd(mol))
    descriptors.append(Descriptors.fr_sulfone(mol))
    descriptors.append(Descriptors.fr_term_acetylene(mol))
    descriptors.append(Descriptors.fr_tetrazole(mol))
    descriptors.append(Descriptors.fr_thiazole(mol))
    descriptors.append(Descriptors.fr_thiocyan(mol))
    descriptors.append(Descriptors.fr_thiophene(mol))
    descriptors.append(Descriptors.fr_unbrch_alkane(mol))
    descriptors.append(Descriptors.fr_urea(mol))
    # MOE-type descriptors
    descriptors.append(Descriptors.LabuteASA(mol))
    descriptors.append(Descriptors.PEOE_VSA1(mol))
    descriptors.append(Descriptors.PEOE_VSA2(mol))
    descriptors.append(Descriptors.PEOE_VSA3(mol))
    descriptors.append(Descriptors.PEOE_VSA4(mol))
    descriptors.append(Descriptors.PEOE_VSA5(mol))
    descriptors.append(Descriptors.PEOE_VSA6(mol))
    descriptors.append(Descriptors.PEOE_VSA7(mol))
    descriptors.append(Descriptors.PEOE_VSA8(mol))
    descriptors.append(Descriptors.PEOE_VSA9(mol))
    descriptors.append(Descriptors.PEOE_VSA10(mol))
    descriptors.append(Descriptors.PEOE_VSA11(mol))
    descriptors.append(Descriptors.PEOE_VSA12(mol))
    descriptors.append(Descriptors.PEOE_VSA13(mol))
    descriptors.append(Descriptors.PEOE_VSA14(mol))
    descriptors.append(Descriptors.SMR_VSA1(mol))
    descriptors.append(Descriptors.SMR_VSA2(mol))
    descriptors.append(Descriptors.SMR_VSA3(mol))
    descriptors.append(Descriptors.SMR_VSA4(mol))
    descriptors.append(Descriptors.SMR_VSA5(mol))
    descriptors.append(Descriptors.SMR_VSA6(mol))
    descriptors.append(Descriptors.SMR_VSA7(mol))
    descriptors.append(Descriptors.SMR_VSA8(mol))
    descriptors.append(Descriptors.SMR_VSA9(mol))
    descriptors.append(Descriptors.SMR_VSA10(mol))
    descriptors.append(Descriptors.SlogP_VSA1(mol))
    descriptors.append(Descriptors.SlogP_VSA2(mol))
    descriptors.append(Descriptors.SlogP_VSA3(mol))
    descriptors.append(Descriptors.SlogP_VSA4(mol))
    descriptors.append(Descriptors.SlogP_VSA5(mol))
    descriptors.append(Descriptors.SlogP_VSA6(mol))
    descriptors.append(Descriptors.SlogP_VSA7(mol))
    descriptors.append(Descriptors.SlogP_VSA8(mol))
    descriptors.append(Descriptors.SlogP_VSA9(mol))
    descriptors.append(Descriptors.SlogP_VSA10(mol))
    descriptors.append(Descriptors.SlogP_VSA11(mol))
    descriptors.append(Descriptors.SlogP_VSA12(mol))
    descriptors.append(Descriptors.EState_VSA1(mol))
    descriptors.append(Descriptors.EState_VSA2(mol))
    descriptors.append(Descriptors.EState_VSA3(mol))
    descriptors.append(Descriptors.EState_VSA4(mol))
    descriptors.append(Descriptors.EState_VSA5(mol))
    descriptors.append(Descriptors.EState_VSA6(mol))
    descriptors.append(Descriptors.EState_VSA7(mol))
    descriptors.append(Descriptors.EState_VSA8(mol))
    descriptors.append(Descriptors.EState_VSA9(mol))
    descriptors.append(Descriptors.EState_VSA10(mol))
    descriptors.append(Descriptors.EState_VSA11(mol))
    descriptors.append(Descriptors.VSA_EState1(mol))
    descriptors.append(Descriptors.VSA_EState2(mol))
    descriptors.append(Descriptors.VSA_EState3(mol))
    descriptors.append(Descriptors.VSA_EState4(mol))
    descriptors.append(Descriptors.VSA_EState5(mol))
    descriptors.append(Descriptors.VSA_EState6(mol))
    descriptors.append(Descriptors.VSA_EState7(mol))
    descriptors.append(Descriptors.VSA_EState8(mol))
    descriptors.append(Descriptors.VSA_EState9(mol))
    descriptors.append(Descriptors.VSA_EState10(mol))   
    # Topological descriptors
    descriptors.append(Descriptors.BalabanJ(mol))
    descriptors.append(Descriptors.BertzCT(mol))
    descriptors.append(Descriptors.HallKierAlpha(mol))
    descriptors.append(Descriptors.Ipc(mol))
    descriptors.append(Descriptors.Kappa1(mol))
    descriptors.append(Descriptors.Kappa2(mol))
    descriptors.append(Descriptors.Kappa3(mol))    
    # Connectivity descriptors
    descriptors.append(Descriptors.Chi0(mol))
    descriptors.append(Descriptors.Chi1(mol))
    descriptors.append(Descriptors.Chi0n(mol))
    descriptors.append(Descriptors.Chi1n(mol))
    descriptors.append(Descriptors.Chi2n(mol))
    descriptors.append(Descriptors.Chi3n(mol))
    descriptors.append(Descriptors.Chi4n(mol))
    descriptors.append(Descriptors.Chi0v(mol))
    descriptors.append(Descriptors.Chi1v(mol))
    descriptors.append(Descriptors.Chi2v(mol))
    descriptors.append(Descriptors.Chi3v(mol))
    descriptors.append(Descriptors.Chi4v(mol))    
    # Other properties
    descriptors.append(Descriptors.qed(mol))
    
    return(descriptors)


# # Compute descriptors on all data first

# In[5]:


# Combine all datasets together (we need to do this for input normalization)

df = pd.concat([df1, df2])
print(df1.shape)
print(df2.shape)
print(df.shape)
# Reset index of df
df = df.reset_index(drop=True)   #VERY IMPORTANT: without reindex iteration has bugs
df.tail(5)


# In[6]:


# Compute descriptors for every sample

newdf = []

for index, row in df.iterrows():

    # Compute descriptors
    smiles_string = df['smiles'][index]
    id_string = df['id'][index]
    mol = Chem.MolFromSmiles(smiles_string)
    descriptors = compute_descriptors(mol, id_string) 
    
    # Append results
    newdf.append(descriptors)


# In[7]:


# Convert descriptors to np array
all_new = np.asarray(newdf)


# In[8]:


all_desc = all_new[:,1:].astype(float)
all_name = all_new[:,:1]
all_desc.shape


# In[9]:


# Is removing missing rows (samples) or columns (descriptors) better?


# In[10]:


# Checking rows
all_desc[~np.isnan(all_desc).any(axis=1)].shape


# In[11]:


# Checking columns
all_desc[:,~np.any(np.isnan(all_desc), axis=0)].shape


# In[12]:


# Removing descriptors with NaN
all_desc = all_desc[:,~np.any(np.isnan(all_desc), axis=0)]


# In[13]:


# Minmax rescale descriptors
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
scaler = MinMaxScaler(feature_range=(0, 1))
all_desc_minmax = scaler.fit_transform(all_desc)


# In[14]:


# Standardize scale descriptors
all_desc_std = StandardScaler().fit_transform(all_desc)


# In[15]:


# Robust scale descriptors
scaler = RobustScaler(quantile_range=(25, 75))
all_desc_robust = scaler.fit_transform(all_desc)


# In[16]:


# Namelist of df for merging
namelist = np.arange(all_desc.shape[1]).tolist()
namelist.insert(0, 'id')


# In[17]:


all_combined = np.concatenate((all_name, all_desc_minmax), axis=1)
final_df = pd.DataFrame(np.asarray(all_combined), columns=namelist)


# In[18]:


final_df.to_csv(homedir+"tox_niehs_desc_minmax.csv", index=False)
final_df.head(5)

