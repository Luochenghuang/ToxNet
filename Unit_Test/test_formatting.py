import sys
sys.path.insert(0, '../chem_scripts') # add file to be tested

import numpy as np
import pandas as pd
import chem_scripts 

def test_cs_load_csv():
    X, y = chem_scripts.cs_load_csv("../data/tox_niehs_int_nontoxic_rdkit.csv")
    try:
        chem_scripts.cs_load_csv(123)
    except(Exception):
        pass
    else:
        raise Exception("Input error. String expected")
    

    assert X.shape == (828,187), 'X shape error'
    assert y.shape == (828,), 'y shape error'
    
    assert type(X) == np.ndarray, 'X type error'
    assert type(y) == np.ndarray, 'y type error'


def test_cs_load_smiles():
    X, y = chem_scripts.cs_load_smiles("../data/tox_niehs_int_nontoxic_smiles.csv")
    try:
        chem_scripts.cs_load_smiles(123)
    except(Exception):
        pass
    else:
        raise Exception("Input error. String expected")
        
    assert X.shape == (828,), 'X shape error'
    assert y.shape == (828,), 'y shape error'
    
    assert type(X) == np.ndarray, 'X type error'
    assert type(y) == np.ndarray, 'y type error'


def test_cs_create_dict():
    characters = set()

    maxlen = 0
    total_lines = 0
    
    X, y = chem_scripts.cs_load_smiles('../data/tox_niehs_tv_verytoxic_smiles.csv')
    X_test, y_test = chem_scripts.cs_load_smiles('../data/tox_niehs_int_verytoxic_smiles.csv')
    
    characters, char_table, char_lookup = chem_scripts.cs_create_dict(X, X_test)
    
    assert len(characters) == 34, 'characters length error'
    assert type(characters) == set, 'characters type error, set expected'
    
    assert len(char_table) == 34, 'characters length error'
    assert type(char_table) == dict, 'characters type error, set expected'
    
    assert len(char_table) == 34, 'characters length error'
    assert type(char_table) == dict, 'characters type error, set expected'
    
    return

def test_cs_prep_data_y():
    X_test, y_test = chem_scripts.cs_load_csv('../data/tox_niehs_int_nontoxic_rdkit.csv')
    y_test, _ = chem_scripts.cs_prep_data_y(y_test, tasktype='classification')
    
    assert y_test.shape == (828, 2), 'shape error, set expected'
    
    assert _ == 2, 'the numbers of classification error,set expected'
    
    return

def test_prep_symbols():
    characters = set()
    maxlen = 0
    total_lines = 0
    X, y = chem_scripts.cs_load_smiles('../data/tox_niehs_int_nontoxic_smiles.csv', smiles_cutoff=250)
    X_sample = X
    characters, maxlen, total_lines = chem_scripts.prep_symbols(X_sample, characters, maxlen, total_lines)
    
    try:
        chem_scripts.prep_symbols(X_sample, "characters", 0, 0)
    except(Exception):
        pass
    else:
        raise Exception("Input type error. Set-int-int-int expected")
        
    assert len(characters) == 31, 'characters length error'
    assert type(characters) == set, 'characters type error, set expected'
    assert maxlen == 174, 'maxlen error'
    assert type(maxlen) == int, 'maxlen type error, integar expected'
    assert total_lines == 828, 'total_lines error'
    assert type(total_lines) == int, 'total_lines type error,integar expected'
    
    return

def test_cs_data_balance():
    X, y = chem_scripts.cs_load_smiles('../data/tox_niehs_int_nontoxic_smiles.csv', smiles_cutoff=250)
    balanced_indices = chem_scripts.cs_data_balance(y)
    
    assert type(balanced_indices) == np.ndarray, 'balanced_indices type error,np.ndarray expected'
    assert balanced_indices.shape == (944,), "balanced_indices shape error, (944,) expected"
    
    return


def test_cs_load_image():
    X, y = chem_scripts.cs_load_image('../data/tox_niehs_int_nontoxic', channel= "engA")
    
    assert X.shape == (821, 25600), "X shape error,(821, 25600) expected"
    assert type(X) == np.ndarray, 'X type error,np.ndarray expected'
    
    assert y.shape == (821,), "y shape error,(821,) expected"
    assert type(y) == np.ndarray, 'y type error,np.ndarray expected'
    
    return