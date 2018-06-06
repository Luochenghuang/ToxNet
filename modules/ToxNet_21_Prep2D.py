
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

from rdkit import Chem


# In[2]:


homedir = os.path.dirname(os.path.realpath('__file__'))
homedir = homedir+"/data/"
archdir = homedir+"/archive/"


# In[3]:


from chem_scripts import cs_compute_features, cs_set_resolution, cs_coords_to_grid, cs_check_grid_boundary
from chem_scripts import cs_channel_mapping, cs_map_atom_to_grid, cs_map_bond_to_grid, cs_grid_to_image


# In[4]:


def gen_image():
    
    exclusion_list = []
    full_array_list = []

    for i in range(0,df.shape[0]):

        # Extract SMILES string
        smiles_string = df["smiles"][i]
        #print(i, smiles_string)

        # Extract ID of molecule
        id_string = df["id"][i]

        # Read SMILES string
        mol = Chem.MolFromSmiles(smiles_string)
        
        # Compute properties
        mol, df_atom, df_bond, nancheckflag = cs_compute_features(mol)
        
        # Intialize grid
        myarray = cs_set_resolution(gridsize, representation=rep)

        # Map coordinates to grid
        df_atom, atomcheckflag = cs_coords_to_grid(df_atom, dim, res)
        
        # Check if outside grid
        sizecheckflag = cs_check_grid_boundary(df_atom, gridsize)

        if sizecheckflag == True or atomcheckflag == True or nancheckflag == True:

            exclusion_list.append(id_string)
            print("EXCLUSION for "+str(id_string))

        else:

            # Initialize channels
            channel = cs_channel_mapping()

            # Map atom to grid
            myarray = cs_map_atom_to_grid(myarray, channel, df_atom, representation=rep)

            # Map bond to grid
            myarray = cs_map_bond_to_grid(myarray, channel, df_atom, df_bond, representation=rep)

            # Visualize status every 1000 steps
            if (i+1)%nskip==0:
                print("*** PROCESSING "+str(i+1)+": "+str(id_string)+" "+str(smiles_string))
                cs_grid_to_image(myarray, mol)

            # Generate combined array of raw input
            curr_array = myarray.flatten()
            curr_array_list = curr_array.tolist()
            full_array_list.append(curr_array_list)

    full_array = np.asarray(full_array_list)
    print(full_array.shape)
    print(exclusion_list)

    return(full_array, exclusion_list)


# # Running image preparation

# In[5]:


dim = 40       # Size of the box in Angstroms, not radius!
res = 0.5      # Resolution of each pixel
rep = "engA"    # Image representation used
nskip = 500    # How many steps till next visualization

gridsize = int(dim/res)


# In[6]:


# Specify dataset name
jobname = "tox_niehs_int"
taskname = ["verytoxic", "nontoxic", "epa", "ghs", "logld50"]

for task in taskname:

    print("PROCESSING TASK: "+str(jobname)+" "+str(task))
    
    # Specify input and output csv
    filein  = homedir+jobname+"_"+task+".csv"
    fileout = homedir+jobname+"_"+task+"_image.csv"
    
    # Specify out npy files
    fileimage = archdir+jobname+"_"+task+"_img_"+rep+".npy" 
    filelabel = archdir+jobname+"_"+task+"_img_label.npy" 
    
    # Generate image
    df = pd.read_csv(filein)
    full_array, exclusion_list = gen_image()
    
    # Dataset statistics before and after image generation
    print("*** Database Specs:")
    print(df.shape[0], len(exclusion_list), int(df.shape[0])-int(len(exclusion_list)))
    
    # Create csv of final data (after exclusion)
    print("*** Separating Database:")
    mod_df = df[~df["id"].isin(exclusion_list)]
    mod_df.to_csv(fileout, index=False)

    # Save generated images as npy
    np.save(fileimage, full_array)
    print(full_array.shape)
    
    # Save labels as npy
    label_array = mod_df[task].as_matrix().astype("float32")
    np.save(filelabel, label_array)
    print(label_array.shape)


# In[7]:


# Specify dataset name
jobname = "tox_niehs_tv"
taskname = ["verytoxic", "nontoxic", "epa", "ghs", "logld50"]

for task in taskname:

    print("PROCESSING TASK: "+str(jobname)+" "+str(task))
    
    # Specify input and output csv
    filein  = homedir+jobname+"_"+task+".csv"
    fileout = homedir+jobname+"_"+task+"_image.csv"
    
    # Specify out npy files
    fileimage = archdir+jobname+"_"+task+"_img_"+rep+".npy" 
    filelabel = archdir+jobname+"_"+task+"_img_label.npy" 
    
    # Generate image
    df = pd.read_csv(filein)
    full_array, exclusion_list = gen_image()
    
    # Dataset statistics before and after image generation
    print("*** Database Specs:")
    print(df.shape[0], len(exclusion_list), int(df.shape[0])-int(len(exclusion_list)))
    
    # Create csv of final data (after exclusion)
    print("*** Separating Database:")
    mod_df = df[~df["id"].isin(exclusion_list)]
    mod_df.to_csv(fileout, index=False)

    # Save generated images as npy
    np.save(fileimage, full_array)
    print(full_array.shape)
    
    # Save labels as npy
    label_array = mod_df[task].as_matrix().astype("float32")
    np.save(filelabel, label_array)
    print(label_array.shape)

