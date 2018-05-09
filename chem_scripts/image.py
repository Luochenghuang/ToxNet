
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


# In[ ]:

# Function to compute atom/bond features

def cs_compute_features(mol):

    AllChem.ComputeGasteigerCharges(mol)
    AllChem.Compute2DCoords(mol)
    # Atomic properties
    df_atom = pd.DataFrame(columns=['idx', 'amu', 'x', 'y', 'z', 'pc', 'hyb', 'val_ex', 'val_im'])
    for atom in mol.GetAtoms():
        # Obtain index
        idx    = atom.GetIdx()
        # Obtain coordinates
        pos = mol.GetConformer(0).GetAtomPosition(idx)
        x = pos.x
        y = pos.y
        z = pos.z
        # Obtain features for image channel augmentation
        amu    = atom.GetAtomicNum()
        pc     = mol.GetAtomWithIdx(idx).GetProp('_GasteigerCharge')
        #fc     = atom.GetFormalCharge()
        hyb    = atom.GetHybridization()
        val_ex = atom.GetExplicitValence()
        val_im = atom.GetImplicitValence()
        #val    = atom.GetTotalValence()
        # Save results in dataframe
        df_atom.loc[len(df_atom)] = [float(idx), float(amu), float(x), float(y), float(z),                                     float(pc), float(hyb), float(val_ex), float(val_im)]
        
    # Bond properties
    df_bond = pd.DataFrame(columns=['atom1', 'atom2', 'btype'])
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        btype = bond.GetBondTypeAsDouble()
        #if bond.GetIsConjugated() == False:
        #    bconj = 0.
        #elif bond.GetIsConjugated() == True:
        #    bconj = 1.
        #else:
        #    raise("ERROR in bond typing")
        #if bond.GetIsAromatic() == False:
        #    barom = 0.
        #elif bond.GetIsAromatic() == True:
        #    barom = 1.
        #else:
        #    raise("ERROR in bond typing")            
        #if bond.IsInRing() == False:
        #    bring = 0.
        #elif bond.IsInRing() == True:
        #    bring = 1.
        #else:
        #    raise("ERROR in bond typing")             
        # Save results in dataframe
        df_bond.loc[len(df_bond)] = [float(atom1), float(atom2), float(btype)]
        
    # Sanity check for NaN
    # Somehow needs to parse in new data to df_atom/df_bond as float if not NaN not detected properly
    if df_atom.isnull().any().any() == True:
        print("WARNING: NaN in df_atom detected! Removing sample from data")
        nancheckflag = True
    elif df_bond.isnull().any().any() == True:
        print("WARNING: NaN in df_bond detected! Removing sample from data")
        nancheckflag = True
    else:
        nancheckflag = False
        
    return(mol, df_atom, df_bond, nancheckflag)


# In[ ]:

# Function to specify dimension & resolution

def cs_set_resolution(gridsize, representation=None, projection="2D"):
    
    # Initalize array for virtual cube.
    # Array element correspond to atomic number (zero means no atom present)

    x_dim = gridsize
    y_dim = gridsize
    z_dim = gridsize
    
    if projection == "2D":
        if representation == "std":
            myarray = np.zeros(shape=[x_dim, y_dim])
        else:
            myarray = np.zeros(shape=[x_dim, y_dim, 4])
    elif projection == "3D":
        if representation == "std":
            myarray = np.zeros(shape=[x_dim, y_dim, z_dim])
        else:
            myarray = np.zeros(shape=[x_dim, y_dim, z_dim, 4])      
    else:
        raise("ERROR: Specify projection")              

    return(myarray)


# In[ ]:

# Function to map coordinates to grid

def cs_coords_to_grid(df_atom, dim, res):
    
    # Assumes atomic coordinates are centered around origin (0,0,0)
    # Need to translate everything to positive coordinate system
    dim = float(dim)
    res = float(res)
    
    # Translate all coordinates by half box size relative to origin
    x_trans = df_atom['x'].values + (dim/2.0)
    y_trans = df_atom['y'].values + (dim/2.0)
    z_trans = df_atom['z'].values + (dim/2.0)
    
    # Rescale coordinates to resolution specified and map to grid
    x_scaled = np.around(x_trans/res)
    df_atom['x_scaled'] = x_scaled
    y_scaled = np.around(y_trans/res)
    df_atom['y_scaled'] = y_scaled
    z_scaled = np.around(z_trans/res)
    df_atom['z_scaled'] = z_scaled
    
    # Sanity check for overlapping atoms
    df_atom['sanity'] = df_atom['x_scaled'].astype(str) + df_atom['y_scaled'].astype(str) +                         df_atom['z_scaled'].astype(str)
    if df_atom['sanity'].nunique() != df_atom['idx'].count():
        print("WARNING: Overlapping atoms detected! Removing sample from data")
        atomcheckflag = True
    else:
        atomcheckflag = False
        
    return(df_atom, atomcheckflag)


# In[ ]:

# Function to check molecule size wrt grid

def cs_check_grid_boundary(df_atom, gridsize):
       
    # Flag is set to "True" if molecule is outside grid
    if (df_atom['x_scaled'] > gridsize-1 ).any() == True:
        sizecheckflag = True
        print("WARNING: Atoms outside grid! Removing sample from data")
    elif (df_atom['y_scaled'] > gridsize-1 ).any() == True:
        sizecheckflag = True
        print("WARNING: Atoms outside grid! Removing sample from data")
    elif (df_atom['z_scaled'] > gridsize-1 ).any() == True:
        sizecheckflag = True
        print("WARNING: Atoms outside grid! Removing sample from data")
    else:
        sizecheckflag = False
        
    return(sizecheckflag)


# In[ ]:

# Function to initialize channel data

def cs_channel_mapping():
    
    channel = {}
    keys = range(0,100)
    values = np.arange(0,100).tolist()
    for i in keys:
        channel[i] = [values[i],0,0,0]
            
    return(channel)


# In[ ]:

# Function to map atom to grid

def cs_map_atom_to_grid(myarray, channel, df_atom, projection="2D", representation="engA"):

    numatom = df_atom['idx'].count()

    for i in range(0,numatom):
        amu_grid = int(df_atom['amu'][i])
        x_grid = int(df_atom['x_scaled'][i])
        y_grid = int(df_atom['y_scaled'][i])
        if representation == "engA": # amu+bond, bond_order, charge, val_ex
            eng_rep = channel[amu_grid]
            eng_rep[2] = float(df_atom['pc'][i])
            eng_rep[3] = float(df_atom['val_ex'][i])
            #print(eng_rep)
            myarray[x_grid, y_grid] = eng_rep
        elif representation == "engB": # amu, bond_order, charge, val_im
            eng_rep = channel[amu_grid]
            eng_rep[2] = float(df_atom['pc'][i])
            eng_rep[3] = float(df_atom['val_im'][i])
            #print(eng_rep)
            myarray[x_grid, y_grid] = eng_rep
        elif representation == "engC": # amu, bond_order, val_ex, val_im
            eng_rep = channel[amu_grid]
            eng_rep[2] = float(df_atom['val_ex'][i])
            eng_rep[3] = float(df_atom['val_im'][i])
            #print(eng_rep)
            myarray[x_grid, y_grid] = eng_rep
        elif representation == "engD": # amu+bond, charge, val_ex, val_im
            eng_rep = channel[amu_grid]
            eng_rep[1] = float(df_atom['pc'][i])
            eng_rep[2] = float(df_atom['val_ex'][i])
            eng_rep[3] = float(df_atom['val_im'][i])
            #print(eng_rep)
            myarray[x_grid, y_grid] = eng_rep
        elif representation == "std":  # amu+bond
            eng_rep = amu_grid
            myarray[x_grid, y_grid] = eng_rep
            
    return(myarray)


# In[ ]:

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


# In[ ]:

def del_atom_on_line2D(ur_drawbond, lx_grid, ly_grid, rx_grid, ry_grid):
    temp_array = ur_drawbond
    for i in range (0,ur_drawbond.shape[0]):
        if ur_drawbond[i][0] == lx_grid:
            if ur_drawbond[i][1] == ly_grid:
                l_atom = i
    for j in range (0,ur_drawbond.shape[0]):
        if ur_drawbond[j][0] == rx_grid:
            if ur_drawbond[j][1] == ry_grid:
                r_atom = j   
    temp_array = np.delete(ur_drawbond, [l_atom, r_atom], axis=0)
    return(temp_array)


# In[ ]:

# Function to map bond to grid

def cs_map_bond_to_grid(myarray, channel, df_atom, df_bond, projection="2D", representation="engA"):

    numatom = df_atom['idx'].count()
    numbond = df_bond['btype'].count()

    for i in range(0,numbond):

        # Intialize atom reference positions
        lx_grid = float('NaN')
        ly_grid = float('NaN')
        rx_grid = float('NaN')
        ry_grid = float('NaN')
        
        # Get index of atoms
        leftatom = int(df_bond['atom1'][i])
        rightatom = int(df_bond['atom2'][i])
        
        # Map atom to grid
        lx_grid = int(df_atom['x_scaled'][leftatom])
        ly_grid = int(df_atom['y_scaled'][leftatom])
        rx_grid = int(df_atom['x_scaled'][rightatom])
        ry_grid = int(df_atom['y_scaled'][rightatom])

        # Draw line between atoms
        num = 30
        bx = np.linspace(lx_grid, rx_grid, num)
        by = np.linspace(ly_grid, ry_grid, num)
        bx_trans = bx[np.newaxis].T
        by_trans = by[np.newaxis].T
            
        # Get coordinates of bond drawn
        drawbond = np.concatenate((bx_trans,by_trans),axis=1)

        # Round to nearest grid
        r_drawbond = np.around(drawbond)
        # Save only unique values
        ur_drawbond = unique_rows(r_drawbond)
        # Remove initial and final coordinates that correspond to atoms
        final_drawbond = del_atom_on_line2D(ur_drawbond, lx_grid, ly_grid, rx_grid, ry_grid)

        # Map bond to channel scheme
        if representation == "engA": # amu, bond_order, charge, val_ex
            eng_rep = channel[0]
            eng_rep[1] = float(df_bond['btype'][i])
            #print(eng_rep)
            bond_color = eng_rep
        elif representation == "engB": # amu, bond_order, charge, val_im
            eng_rep = channel[0]
            eng_rep[1] = float(df_bond['btype'][i])
            #print(eng_rep)
            bond_color = eng_rep
        elif representation == "engC": # amu, bond_order, val_ex, val_im
            eng_rep = channel[0]
            eng_rep[1] = float(df_bond['btype'][i])
            #print(eng_rep)
            bond_color = eng_rep
        elif representation == "engD": # amu, charge, val_ex, val_im
            eng_rep = channel[2]     # Yes I know it's He, but it's not used
            #print(eng_rep)
            bond_color = eng_rep
        elif representation == "std":  # amu
            eng_rep = int(2)     # Yes I know it's He, but it's not used
            #print(eng_rep)
            bond_color = eng_rep            

        # Draw bond on grid
        numpixel = final_drawbond.shape[0]
        if numpixel > 0:
            for k in range(0,numpixel):
                bx_grid = int(final_drawbond[k][0]) #Should we "integer-ize here or a step before?
                by_grid = int(final_drawbond[k][1])
                myarray[bx_grid, by_grid] = bond_color
        else:
            myarray = myarray
                    
    return(myarray)


# In[ ]:

# Function to display grid-image vs RDKit image
from IPython.display import Image, display
from IPython.display import display
from scipy.misc import imsave

def cs_grid_to_image (myarray, mol):
    file1 = "temp1.png"
    file2 = "temp2.png"
    imsave(file1, myarray)
    Draw.MolToFile(mol, file2)
    x = Image(filename=file1, width=200) 
    y = Image(filename=file2, width=200)
    display(x,y)


# In[ ]:




# In[ ]:




# In[ ]:



