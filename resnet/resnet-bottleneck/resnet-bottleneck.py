import numpy as np

def s(x, Ws):
    return x @ np.array(Ws) if Ws else x

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    
    x = np.array(x)
    W1, W2, W3 = np.array(W1), np.array(W2), np.array(W3)

    output_ = np.maximum(0, x @ W1)
    output__ = np.maximum(0, output_ @ W2)
    
    output = np.maximum(0, output__ @ W3 + s(x, Ws))
    return output