import pandas as pd
import numpy as np
import chem_scripts 

def test_cs_prep_data_y():
    X_test, y_test = chem_scripts.cs_load_csv('tox_niehs_int_nontoxic_rdkit.csv')
    y_test, _ = chem_scripts.cs_prep_data_y(y_test, tasktype='classification')
    
    assert y_test.shape == (828, 2), 'shape error, set expected'
    
    assert _ == 2, 'the numbers of classification error,set expected'
    
    return
 