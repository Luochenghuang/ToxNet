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