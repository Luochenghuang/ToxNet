import pandas as pd
import numpy as np
import chem_scripts

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
