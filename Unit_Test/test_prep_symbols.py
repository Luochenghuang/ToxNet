import pandas as pd
import numpy as np
import chem_scripts 

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