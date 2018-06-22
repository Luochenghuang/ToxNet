import numpy as np
import pandas as pd
import chem_scripts 

def test_cs_auc():
    X_test, y_test = chem_scripts.cs_load_smiles('tox_niehs_int_verytoxic_smiles.csv')
    y_tmp = y_test
    auc_train = chem_scripts.cs_auc(y_tmp, y_tmp)
    
    assert auc_train == 1,'auc calculation error'
    assert type(y_tmp) == np.ndarray, 'input dataset type error, set expected'
    
    return