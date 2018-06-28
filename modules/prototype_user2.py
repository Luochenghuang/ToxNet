
# coding: utf-8

# In[5]:


import os
import numpy as np
import pandas as pd
import sys

from keras.metrics import categorical_accuracy, binary_accuracy
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import StratifiedKFold, KFold

# # Loading Data

# In[8]:
from chem_scripts import cs_load_csv, cs_load_smiles, cs_create_dict, cs_prep_data_X, cs_prep_data_y, cs_data_balance
from chem_scripts import cs_setup_mlp, cs_setup_rnn

homedir = os.path.dirname(os.path.realpath('__file__'))+"/../data/"


def f_nn(taskname, network_name, df_test):
    tasktype='classification'
    jobname = "tox_niehs"

    # Specify location of data
    
    # Load data
    if network_name == 'mlp':
        filename=homedir+jobname+"_tv_"+taskname+"_rdkit.csv"
        X, y = cs_load_csv(filename)

    
    # setting
    params = {"dropval":0.5, "num_layer":2, "relu_type":"prelu",
              "layer1_units":64, "layer2_units":64, "layer3_units":64,
              "layer4_units":64, "layer5_units":64, "layer6_units":64,
              "reg_type": "l2", "reg_val": 2.5 }
    prototype = True
    if network_name == 'mlp':
        if taskname == 'nontoxic':
            params['layer1_units'] = 256
            params['layer2_units'] = 32
        elif taskname == 'verytoxic':
            params['num_layer'] = 3
            params['layer1_units'] = 128
            params['layer2_units'] = 256
        elif taskname == 'ghs':
            params['layer2_units'] = 16
            params['reg_val'] = 2
        elif taskname == 'epa':
            params['layer1_units'] = 256
            params['layer2_units'] = 16
            params['relu_type'] = 'relu'
            params['reg_val'] = 4.2
        else:
            tasktype = 'regression'
            params['relu_type'] = 'relu'
            params['layer2_units'] = 256
            params['reg_val'] = 2.6

    if tasktype == "classification":
    
        # Do class-balancing
        balanced_indices = cs_data_balance(y)
        X = X[balanced_indices]
        y = y[balanced_indices]
        
        # One-hot encoding
        y, y_class = cs_prep_data_y(y, tasktype=tasktype)
        
    elif tasktype == "regression":
        y_class = 1

    # Setup network
    if network_name == 'mlp':
        model, submodel = cs_setup_mlp(params, inshape=X.shape[1], classes=y_class)

    # Setup callbacks
    filecp = jobname+"_"+network_name+'_'+taskname+"_bestweights_trial_1_0.hdf5"
    
    # Reload best model & compute results
    model.load_weights(homedir+'../result/'+network_name+'/'+filecp)
    y_preds_result = model.predict(df_test)
    return y_preds_result

''' 
########################## Used For Test ONLY !!!! ##################################
################ RNN can be used with GPU and CUDNN registered ######################
task='nontoxic'
df = pd.read_csv("../data/tox_niehs_int_"+task+"_rdkit.csv").drop(columns=[task,'id'])
df_test = pd.DataFrame()
df_test = df_test.append(df.iloc[50,:], sort=False)
y = f_nn(task, 'rnn', df_test)
print(y)
'''