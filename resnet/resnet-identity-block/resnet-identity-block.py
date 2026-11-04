import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """

    x, W1, W2 = np.array(x), np.array(W1), np.array(W2)
    
    hidden_output = np.maximum(0, x @ W1.swapaxes(-2, -1))
    output = np.maximum(0, hidden_output @ W2.swapaxes(-2, -1) + x)
    return output.tolist()