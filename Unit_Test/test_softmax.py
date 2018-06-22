import numpy as np

def test_softmax():
    K = np.zeros((1,5))
    Y = softmax(K)
    
    assert len(Y) == 1, 'length error'
    assert type(Y) == array, 'type error, set expected'
    assert Y[0,:] == [0.2, 0.2, 0.2, 0.2, 0.2], 'calculation error,set expected'
    
    return
