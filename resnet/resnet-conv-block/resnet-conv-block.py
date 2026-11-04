import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """

    x, W1, W2, Ws = np.array(x), np.array(W1), np.array(W2), np.array(Ws)
    
    hidden_output = np.maximum(
        0,
        x @ W1
    )
    output = np.maximum(
        0,
        hidden_output @ W2 + x @ Ws
    )
    return output.tolist()