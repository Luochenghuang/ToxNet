import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.metrics import roc_auc_score
from itertools import combinations
from keras.utils import np_utils
from keras.preprocessing import sequence
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Input, Masking, add, concatenate
from keras.layers import Embedding, GRU, LSTM, CuDNNGRU, CuDNNLSTM, TimeDistributed, Bidirectional
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras.regularizers import l2, l1, l1_l2
from keras import backend as K

def softmax(array):
    exp = np.exp(array)
    totals = np.sum(exp, axis=1)
    for i in range(len(exp)):
        exp[i, :] /= totals[i]
    return exp


# In[ ]:

def cs_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc


# In[ ]:

def cs_multiclass_auc(y_true, y_pred):
    n = y_pred.shape[1]
    auc_dict = dict()
    for pair in combinations(range(n), 2):
        subset = [i for i in range(len(y_true)) if 1 in [y_true[i, pair[0]], y_true[i, pair[1]]]]
        y_true_temp = y_true[subset]
        y_pred_temp = y_pred[subset]
        y_pred_temp = y_pred_temp[:, [pair[0], pair[1]]]
        y_pred_temp = softmax(y_pred_temp)
        auc_dict[pair] = roc_auc_score(y_true_temp[:, pair[1]], y_pred_temp[:, 1])
    total = 0.0
    for key in auc_dict.keys():
        total += auc_dict[key]
    total /= len(list(combinations(range(n), 2)))
    return total


# In[ ]:

def cs_compute_results(model, classes=None, train_data=None, valid_data=None, test_data=None, df_out=None):
    
    # Evaluate results on training set
    X_tmp = train_data[0]
    y_tmp = train_data[1]    
    loss_train = model.evaluate(X_tmp, y_tmp, verbose=0)
    
    if classes == 1:
        rmse_train = np.sqrt(loss_train)
    elif classes == 2:
        y_preds_train = model.predict(X_tmp)
        auc_train = cs_auc(y_tmp, y_preds_train)
    elif classes > 2:
        y_preds_train = model.predict(X_tmp)
        auc_train = cs_multiclass_auc(y_tmp, y_preds_train)
    else:
        raise(Exception('Error in determine problem type'))
        
    # Evaluate results on validation set    
    X_tmp = valid_data[0]
    y_tmp = valid_data[1]
    loss_valid = model.evaluate(X_tmp, y_tmp, verbose=0)
    if classes == 1:
        rmse_valid = np.sqrt(loss_valid)
    elif classes == 2:
        y_preds_valid = model.predict(X_tmp)
        auc_valid = cs_auc(y_tmp, y_preds_valid)
    elif classes > 2:
        y_preds_valid = model.predict(X_tmp)
        auc_valid = cs_multiclass_auc(y_tmp, y_preds_valid)
    else:
        raise(Exception('Error in determine problem type'))
    
    # Evaluate results on test set
    X_tmp = test_data[0]
    y_tmp = test_data[1]    
    loss_test = model.evaluate(X_tmp, y_tmp, verbose=0)
    if classes == 1:
        y_preds_test = model.predict(X_tmp)
        rmse_test = np.sqrt(loss_test)
    elif classes == 2:
        y_preds_test = model.predict(X_tmp)
        auc_test = cs_auc(y_tmp, y_preds_test)
    elif classes > 2:
        y_preds_test = model.predict(X_tmp)
        auc_test = cs_multiclass_auc(y_tmp, y_preds_test)
    else:
        raise(Exception('Error in determine problem type'))
    
    if classes == 1:
        print("\nFINAL TRA_LOSS: %.3f"%(loss_train))
        print("FINAL VAL_LOSS: %.3f"%(loss_valid))
        print("FINAL TST_LOSS: %.3f"%(loss_test))
        print("FINAL TRA_RMSE: %.3f"%(rmse_train))
        print("FINAL VAL_RMSE: %.3f"%(rmse_valid))
        print("FINAL TST_RMSE: %.3f"%(rmse_test))
        df_out.loc[len(df_out)] = [loss_train, loss_valid, loss_test, rmse_train, rmse_valid, rmse_test]
    else:
        print("\nFINAL TRA_LOSS: %.3f"%(loss_train))
        print("FINAL VAL_LOSS: %.3f"%(loss_valid))
        print("FINAL TST_LOSS: %.3f"%(loss_test))
        print("FINAL TRA_AUC: %.3f"%(auc_train))
        print("FINAL VAL_AUC: %.3f"%(auc_valid))
        print("FINAL TST_AUC: %.3f"%(auc_test))
        df_out.loc[len(df_out)] = [loss_train, loss_valid, loss_test, auc_train, auc_valid, auc_test]
    return y_preds_test


# In[ ]:

def cs_keras_to_seaborn(history):
    tmp_frame = pd.DataFrame(history.history)
    keys = list(history.history.keys())
    features = [x for x in keys if "val_" not in x and "val_" + x in keys]
    cols = ['epoch', 'phase'] + features
    output_df = pd.DataFrame(columns=cols)
    epoch = 1
    for i in range(len(tmp_frame)):
        new_row = [epoch, 'train'] + [tmp_frame.loc[i, f] for f in features]
        output_df.loc[len(output_df)] = new_row
        new_row = [epoch, 'validation'] + [tmp_frame.loc[i, "val_" + f] for f in features]
        output_df.loc[len(output_df)] = new_row
        epoch += 1
    return output_df


# In[ ]:

#def cs_make_plots(hist_df, filename=None):
    #fig, axes = plt.subplots(1, 1)
    #sns.pointplot(x='epoch', y='loss', hue='phase', data=hist_df, ax=axes)
    #axes.set_title('Loss Curve', fontdict={'size': 20})
    #axes.set_ylim(np.min(hist_df['loss']), np.max(hist_df['loss']))
    #plt.show()

# In[ ]:

# Function for loading MLP data

def cs_load_csv(filename):
        
    data = pd.read_csv(filename)
    np_data = np.asarray(data)
    
    # Assumes first column is label
    y = np_data[:,1]
    X = np_data[:,2:]
    
    # Free memory
    np_data = np.asarray([])
    del data
    
    return(X, y)


# In[ ]:

# Function for loading SMILES data

def cs_load_smiles(filename, smiles_cutoff=250):
    
    # Specify cutoff length of SMILES
    
    data = pd.read_csv(filename)
    X = data["smiles"].values
    
    # Assumes last column is label
    y = np.asarray(data.ix[:,-1])    

    # Free memory
    del data
    
    return(X, y)    


# In[ ]:

# Function for loading SMILES data

def cs_load_image(filename, channel="std"):
    
    newfile = filename+"_img_"+channel+".npy" 
    X = np.load(newfile)
    newfile = filename+"_img_label.npy" 
    y = np.load(newfile)

    return(X, y)    


# In[ ]:

# Function for processing characters to dictionary mapping

def prep_symbols(X_sample, characters, maxlen, total_lines):
    
    for line in X_sample:
        total_lines += 1
        if len(line) > maxlen:
            maxlen = len(line)
        for c in line:
            characters.add(c)
        
    return(characters, maxlen, total_lines)

def cs_create_dict(X, X_test):
    
    # Create a set of all symbols used in the sequences
    characters = set()

    maxlen = 0
    total_lines = 0

    characters, maxlen, total_lines = prep_symbols(X, characters, maxlen, total_lines)    
    characters, maxlen, total_lines = prep_symbols(X_test, characters, maxlen, total_lines)      
            
    # Make a table of different elements of SMILES
    char_table = dict()
    char_lookup = dict()
    i = 1
    for c in characters:
        char_table[c] = i
        char_lookup[i] = c
        i += 1

    print("Character mapping: "+str(char_table))
    
    return(characters, char_table, char_lookup)


# In[ ]:

# Function for mapping chars to integers

def cs_prep_data_X(X_sample, datarep=None, char_table=None, smiles_cutoff=250):
    
    if datarep == "text":
        
        X_new = list()
        i = 0
        for line in X_sample:
            X_new.append([ char_table[c] for c in line ])
            i += 1
            
        # Apply zero padding
        X_sample = sequence.pad_sequences(X_new,dtype='int32', padding='post',value=0, maxlen=smiles_cutoff-10)
        X_sample = sequence.pad_sequences(X_sample,dtype='int32', padding='pre',value=0, maxlen=smiles_cutoff)
        X_new = list()
            
    return(X_sample)


# In[ ]:

# Function for performing class balancing

def cs_data_balance(class_list):
    
    # Count how many samples for each class is present
    counts = Counter(class_list)
    
    # Determine max class and count
    maxclass, maxcount = Counter(class_list).most_common(1)[0]

    # Construct separate list of each class to match max class and concat to single list
    index_lists = []
    for key in counts.keys():
        tmp_list = [i for i in range(len(class_list)) if class_list[i] == key]
        index_lists.append(tmp_list)
        # Oversample non-max class until max count is reached
        if len(tmp_list) < maxcount:
            index_lists.append(np.random.choice(tmp_list, size=maxcount-len(tmp_list)))#, replace=True))
    index_list = np.concatenate(index_lists)
    np.random.shuffle(index_list)
    
    return index_list


# In[ ]:

# Function for one-hot encoding

def cs_prep_data_y(y_sample, tasktype=None):

    if tasktype == "regression":
    
        # Ensure/convert to float
        y_sample = y_sample.astype("float32")
        print("y dim: "+str(y_sample.shape))
        
        return(y_sample)
        
    elif tasktype == "classification":
        
        # Ensure/convert to integers
        y_sample = y_sample.astype("int32")

        # One-hot encode outputs
        y_sample = np_utils.to_categorical(y_sample)
        y_class = y_sample.shape[1]

        print("y dim: "+str(y_sample.shape))
        print("y no. class: "+str(y_class))
    
        return (y_sample, y_class)


# In[ ]:

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

def cs_setup_mlp(params, inshape=None, classes=None):
    
    # Parse network hyperparameters
    num_layer = int(params['num_layer'])
    units1 = int(params['layer1_units'])
    units2 = int(params['layer2_units'])
    units3 = int(params['layer3_units'])
    units4 = int(params['layer4_units'])
    units5 = int(params['layer5_units'])
    units6 = int(params['layer6_units'])
    relu_flag = str(params['relu_type'])
    dropval = float(params['dropval'])
    reg_flag = str(params['reg_type'])
    reg_val = 10**(-float(params['reg_val']))
    
    # Setup regularizer
    if reg_flag == "l1":
        reg = l1(reg_val)
        print("Regularizer "+reg_flag+" set at "+str(reg_val))
    elif reg_flag == "l2":
        reg = l2(reg_val)
        print("Regularizer "+reg_flag+" set at "+str(reg_val))
    elif reg_flag == "l1_l2":
        reg = l1_l2(reg_val)
        print("Regularizer "+reg_flag+" set at "+str(reg_val))
    else:
        reg = None
        print("NOTE: No regularizers used")
        
    # Setup neural network
    inlayer = Input(shape=[inshape])
    if num_layer >= 1:
        x = Dense(units1, kernel_regularizer=reg)(inlayer)
        if relu_flag == "relu":
            x = Activation("relu")(x)
        elif relu_flag == "elu":
            x = Activation("elu")(x)
        elif relu_flag == "prelu":
            x = PReLU()(x)
        elif relu_flag == "leakyrelu":
            x = LeakyReLU()(x)
        x = Dropout(dropval)(x)
    if num_layer >= 2:
        x = Dense(units2, kernel_regularizer=reg)(x)
        if relu_flag == "relu":
            x = Activation("relu")(x)
        elif relu_flag == "elu":
            x = Activation("elu")(x)
        elif relu_flag == "prelu":
            x = PReLU()(x)
        elif relu_flag == "leakyrelu":
            x = LeakyReLU()(x)
        x = Dropout(dropval)(x)
    if num_layer >= 3:
        x = Dense(units3, kernel_regularizer=reg)(x)
        if relu_flag == "relu":
            x = Activation("relu")(x)
        elif relu_flag == "elu":
            x = Activation("elu")(x)
        elif relu_flag == "prelu":
            x = PReLU()(x)
        elif relu_flag == "leakyrelu":
            x = LeakyReLU()(x)
        x = Dropout(dropval)(x)
    if num_layer >= 4:
        x = Dense(units4, kernel_regularizer=reg)(x)
        if relu_flag == "relu":
            x = Activation("relu")(x)
        elif relu_flag == "elu":
            x = Activation("elu")(x)
        elif relu_flag == "prelu":
            x = PReLU()(x)
        elif relu_flag == "leakyrelu":
            x = LeakyReLU()(x)
        x = Dropout(dropval)(x)
    if num_layer >= 5:
        x = Dense(units5, kernel_regularizer=reg)(x)
        if relu_flag == "relu":
            x = Activation("relu")(x)
        elif relu_flag == "elu":
            x = Activation("elu")(x)
        elif relu_flag == "prelu":
            x = PReLU()(x)
        elif relu_flag == "leakyrelu":
            x = LeakyReLU()(x)
        x = Dropout(dropval)(x)
    if num_layer >= 6:
        x = Dense(units6, kernel_regularizer=reg)(x)
        if relu_flag == "relu":
            x = Activation("relu")(x)
        elif relu_flag == "elu":
            x = Activation("elu")(x)
        elif relu_flag == "prelu":
            x = PReLU()(x)
        elif relu_flag == "leakyrelu":
            x = LeakyReLU()(x)
        x = Dropout(dropval)(x)
    
    # Specify output layer
    if classes == 1:
        label = Dense(classes, activation='linear', name='predictions')(x)
    elif classes >= 2:
        label = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        raise("ERROR in specifying tasktype")
        
    # Create base model
    model = Model(inputs=inlayer,outputs=label, name='MLP')
    
    # Create intermediate model
    submodel = Model(inputs=inlayer,outputs=x, name='MLP_truncated')
    
    # Specify training method
    if classes == 1:
        model.compile(optimizer="RMSprop", loss="mean_squared_error")
        submodel.compile(optimizer="RMSprop", loss="mean_squared_error")
    elif classes >= 2:
        model.compile(optimizer="RMSprop", loss="categorical_crossentropy")
        submodel.compile(optimizer="RMSprop", loss="categorical_crossentropy")
    else:
        raise("ERROR in specifying tasktype")
    
    return(model, submodel)


# # RNN network designs

# In[ ]:

def cs_setup_rnn(params, inshape=None, classes=None, char=None):
    
    # Parse network hyperparameters
    em_dim = int(params['em_dim']*10)
    kernel_size = 3
    filters = int(params['conv_units']*6)    
    num_layer = int(params['num_layer'])
    units1 = int(params['layer1_units']*6)
    units2 = int(params['layer2_units']*6)
    units3 = int(params['layer3_units']*6)
    relu_flag = str(params['relu_type'])
    dropval = float(params['dropval'])
    reg_flag = str(params['reg_type'])
    reg_val = 10**(-float(params['reg_val']))

    # Setup regularizer
    if reg_flag == "l1":
        reg = l1(reg_val)
        print("Regularizer "+reg_flag+" set at "+str(reg_val))
    elif reg_flag == "l2":
        reg = l2(reg_val)
        print("Regularizer "+reg_flag+" set at "+str(reg_val))
    elif reg_flag == "l1_l2":
        reg = l1_l2(reg_val)
        print("Regularizer "+reg_flag+" set at "+str(reg_val))
    else:
        reg = None
        print("NOTE: No regularizers used")
    
    # Setup neural network
    inlayer = Input(shape=[inshape])
    x = Embedding(input_dim=len(char)+1,output_dim=em_dim)(inlayer)   
    x = Conv1D(filters, kernel_size, strides=1, padding="same", kernel_regularizer=reg)(x)
    if relu_flag == "relu":
        x = Activation("relu")(x)
    elif relu_flag == "elu":
        x = Activation("elu")(x)
    elif relu_flag == "prelu":
        x = PReLU()(x)
    elif relu_flag == "leakyrelu":
        x = LeakyReLU()(x)
    if params['celltype'] == "GRU":
        if num_layer == 1:
            x = Bidirectional(CuDNNGRU(units1, return_sequences=False))(x)
            x = Dropout(dropval)(x)
        elif num_layer == 2:
            x = Bidirectional(CuDNNGRU(units1, return_sequences=True))(x)
            x = Dropout(dropval)(x)
            x = Bidirectional(CuDNNGRU(units2, return_sequences=False))(x)
            x = Dropout(dropval)(x)
        elif num_layer == 3:
            x = Bidirectional(CuDNNGRU(units1, return_sequences=True))(x)
            x = Dropout(dropval)(x)
            x = Bidirectional(CuDNNGRU(units2, return_sequences=True))(x)
            x = Dropout(dropval)(x)
            x = Bidirectional(CuDNNGRU(units3, return_sequences=False))(x)
            x = Dropout(dropval)(x)
    if params['celltype'] == "LSTM":
        if num_layer == 1:
            x = Bidirectional(CuDNNLSTM(units1, return_sequences=False))(x)
            x = Dropout(dropval)(x)
        elif num_layer == 2:
            x = Bidirectional(CuDNNLSTM(units1, return_sequences=True))(x)
            x = Dropout(dropval)(x)
            x = Bidirectional(CuDNNLSTM(units2, return_sequences=False))(x)
            x = Dropout(dropval)(x)
        elif num_layer == 3:
            x = Bidirectional(CuDNNLSTM(units1, return_sequences=True))(x)
            x = Dropout(dropval)(x)
            x = Bidirectional(CuDNNLSTM(units2, return_sequences=True))(x)
            x = Dropout(dropval)(x)
            x = Bidirectional(CuDNNLSTM(units3, return_sequences=False))(x)
            x = Dropout(dropval)(x)
    
    # Specify output layer
    if classes == 1:
        label = Dense(classes, activation='linear', name='predictions')(x)
    elif classes >= 2:
        label = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        raise("ERROR in specifying tasktype")
        
    # Create base model
    model = Model(inputs=inlayer,outputs=label, name='SMILES2vec')
    
    # Create intermediate model
    submodel = Model(inputs=inlayer,outputs=x, name='SMILES2vec_truncated')
    
    # Specify training method
    if classes == 1:
        model.compile(optimizer="RMSprop", loss="mean_squared_error")
        submodel.compile(optimizer="RMSprop", loss="mean_squared_error")
    elif classes >= 2:
        model.compile(optimizer="RMSprop", loss="categorical_crossentropy")
        submodel.compile(optimizer="RMSprop", loss="categorical_crossentropy")
    else:
        raise("ERROR in specifying tasktype")
    
    return(model, submodel)


# # CNN Network Designs

# In[ ]:

def conv2d_bn(x, nb_filter, kernel_size=4, padding='same', strides=2):

    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1
        
    x = Conv2D(nb_filter, kernel_size=(kernel_size,kernel_size), strides=(strides,strides), padding=padding)(x)
    x = Activation("relu")(x)
    
    return x


# In[ ]:

def inception_resnet_v2_A(input_tensor, nb_params, last_params, scale_residual=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input_tensor

    ir1 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)

    ir2 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    ir2 = Conv2D(nb_params, kernel_size=(3,3), activation='relu', padding='same')(ir2)

    ir3 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    ir3 = Conv2D(int(nb_params*1.5), kernel_size=(3,3), activation='relu', padding='same')(ir3)
    ir3 = Conv2D(int(nb_params*2.0), kernel_size=(3,3), activation='relu', padding='same')(ir3)

    ir_merge = concatenate([ir1, ir2, ir3], axis=channel_axis)

    ir_conv = Conv2D(last_params, kernel_size=(1,1), activation='linear', padding='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = Activation("relu")(out)
    
    return out


# In[ ]:

def inception_resnet_v2_B(input_tensor, nb_params, last_params, scale_residual=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input_tensor

    ir1 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)

    ir2 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    ir2 = Conv2D(int(nb_params*1.25), kernel_size=(1,7), activation='relu', padding='same')(ir2)
    ir2 = Conv2D(int(nb_params*1.50), kernel_size=(7,1), activation='relu', padding='same')(ir2)

    ir_merge = concatenate([ir1, ir2], axis=channel_axis)

    ir_conv = Conv2D(last_params, kernel_size=(1,1), activation='linear', padding='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = Activation("relu")(out)
    
    return out


# In[ ]:

def inception_resnet_v2_C(input_tensor, nb_params, last_params, scale_residual=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input_tensor

    ir1 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)

    ir2 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    ir2 = Conv2D(int(nb_params*1.1666666), kernel_size=(1,3), activation='relu', padding='same')(ir2)
    ir2 = Conv2D(int(nb_params*1.3333333), kernel_size=(3,1), activation='relu', padding='same')(ir2)

    ir_merge = concatenate([ir1, ir2], axis=channel_axis)

    ir_conv = Conv2D(last_params, kernel_size=(1,1), activation='linear', padding='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = Activation("relu")(out)
    
    return out


# In[ ]:

def reduction_A(input_tensor, nb_params):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling2D((3,3), padding='valid', strides=(2,2))(input_tensor)

    r2 = Conv2D(int(nb_params*1.5), kernel_size=(3,3), activation='relu', padding='valid', strides=(2,2))(input_tensor)

    r3 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    r3 = Conv2D(nb_params, kernel_size=(3,3), activation='relu', padding='same')(r3)
    r3 = Conv2D(int(nb_params*1.5), kernel_size=(3,3), activation='relu', padding='valid', strides=(2,2))(r3)

    m = concatenate([r1, r2, r3], axis=channel_axis)
    m = Activation('relu')(m)
    
    return m


# In[ ]:

def reduction_resnet_v2_B(input_tensor, nb_params):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling2D((3,3), padding='valid', strides=(2,2))(input_tensor)

    r2 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    r2 = Conv2D(int(nb_params*1.5), kernel_size=(3,3), activation='relu', padding='valid', strides=(2,2))(r2)

    r3 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    r3 = Conv2D(int(nb_params*1.125), kernel_size=(3,3), activation='relu', padding='valid', strides=(2, 2))(r3)

    r4 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    r4 = Conv2D(int(nb_params*1.125), kernel_size=(3,3), activation='relu', padding='same')(r4)
    r4 = Conv2D(int(nb_params*1.25), kernel_size=(3,3), activation='relu', padding='valid', strides=(2, 2))(r4)
    
    m = concatenate([r1, r2, r3, r4], axis=channel_axis)
    m = Activation('relu')(m)
    
    return m


# In[ ]:

def end_block_droppool(input_tensor, dropval):
        
    x = GlobalAveragePooling2D(data_format="channels_last", name="final_pool")(input_tensor)
    x = Dropout(dropval, name="dropout_end")(x)
    
    return(input_tensor, x)


# In[ ]:

def end_block_pool(input_tensor):
    
    x = GlobalAveragePooling2D(data_format="channels_last", name="final_pool")(input_tensor)
    
    return(input_tensor, x)


# In[ ]:

def cs_setup_cnn(params, inshape=None, classes=None):
    """Instantiate the Inception v3 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `tf` dim ordering)
            or `(3, 299, 299)` (with `th` dim ordering).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.
    """
    
    #Clear GPU memory
    K.clear_session()
       
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
    else:
        channel_axis = -1
    print("Channel axis is "+str(channel_axis))
        
    # Assign image input
    inlayer = Input(inshape)
    x = conv2d_bn(inlayer, params['conv1_units'], kernel_size=4, strides=2)

    # Inception Resnet A
    for i in range(params['num_block1']):
        last_params = params['conv1_units']
        x = inception_resnet_v2_A(x, params['conv2_units'], last_params, scale_residual=False)

    # Reduction A
    x = reduction_A(x, params['conv3_units'])

    # Inception Resnet B
    for i in range(params['num_block2']):
        last_params = int(params['conv1_units']+(params['conv3_units']*3))
        x = inception_resnet_v2_B(x, params['conv4_units'], last_params, scale_residual=False)

    # Reduction Resnet B
    x = reduction_resnet_v2_B(x, params['conv5_units'])

    # Inception Resnet C
    for i in range(params['num_block3']):
        last_params = int(params['conv1_units']+(params['conv3_units']*3))+int(params['conv5_units']*3.875)
        x = inception_resnet_v2_C(x, params['conv6_units'], last_params, scale_residual=False)
            
    # Classification block
    before_pool, after_pool = end_block_droppool(x, params['dropval'])

    # Specify output layer
    if classes == 1:
        label = Dense(classes, activation='linear', name='predictions')(after_pool)
    elif classes >= 2:
        label = Dense(classes, activation='softmax', name='predictions')(after_pool)
    else:
        raise("ERROR in specifying tasktype")
        
    # Create base model
    model = Model(inputs=inlayer,outputs=label, name='Chemception')
    
    # Create intermediate model
    submodel = Model(inputs=inlayer,outputs=after_pool, name='Chemception_truncated')
    
    # Specify training method
    if classes == 1:
        model.compile(optimizer="RMSprop", loss="mean_squared_error")
        submodel.compile(optimizer="RMSprop", loss="mean_squared_error")
    elif classes >= 2:
        model.compile(optimizer="RMSprop", loss="categorical_crossentropy")
        submodel.compile(optimizer="RMSprop", loss="categorical_crossentropy")
    else:
        raise("ERROR in specifying tasktype")
    
    return(model, submodel)
