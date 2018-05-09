
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd

from keras.utils import np_utils
from keras.preprocessing import sequence
from collections import Counter


# In[ ]:

# Function for loading MLP data

def cs_load_csv(filename):
        
    data = pd.read_csv(filename)
    np_data = np.asarray(data)
    
    # Assumes first column is label
    y = np_data[:,0]
    X = np_data[:,1:]
    
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



