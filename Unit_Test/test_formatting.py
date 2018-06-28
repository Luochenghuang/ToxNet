import numpy as np
import pandas as pd
import chem_scripts 

def test_cs_load_csv():
    chem_scripts.cs_load_csv("tox_niehs_int_nontoxic_rdkit.csv")
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
    
    return

def test_cs_load_smiles():
    chem_scripts.cs_load_smiles("tox_niehs_int_nontoxic_smiles.csv")
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
    
    return

def test_cs_create_dict():
    characters = set()

    maxlen = 0
    total_lines = 0
    
    X, y = chem_scripts.cs_load_smiles('tox_niehs_tv_verytoxic_smiles.csv')
    X_test, y_test = chem_scripts.cs_load_smiles('tox_niehs_int_verytoxic_smiles.csv')
    
    characters, char_table, char_lookup = chem_scripts.cs_create_dict(X, X_test)
    
    assert len(characters) == 34, 'characters length error'
    assert type(characters) == set, 'characters type error, set expected'
    
    assert len(char_table) == 34, 'characters length error'
    assert type(char_table) == dict, 'characters type error, set expected'
    
    assert len(char_table) == 34, 'characters length error'
    assert type(char_table) == dict, 'characters type error, set expected'
    
    return

def test_cs_prep_data_y():
    X_test, y_test = chem_scripts.cs_load_csv('tox_niehs_int_nontoxic_rdkit.csv')
    y_test, _ = chem_scripts.cs_prep_data_y(y_test, tasktype='classification')
    
    assert y_test.shape == (828, 2), 'shape error, set expected'
    
    assert _ == 2, 'the numbers of classification error,set expected'
    
    return

def test_prep_symbols():
    characters = set()
    maxlen = 0
    total_lines = 0
    X, y = chem_scripts.cs_load_smiles('tox_niehs_int_nontoxic_smiles.csv', smiles_cutoff=250)
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