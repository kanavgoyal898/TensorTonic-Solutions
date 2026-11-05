import numpy as np

def conv(x, W):
    return x @ W

def batch_norm(x, gamma, beta, eps=1e-5):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    x_cap =  (x - mu)/np.sqrt(sigma**2 + eps)
    return gamma * x_cap + beta

def relu(x):
    return np.maximum(0, x)

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """

    x, W1, W2 = np.array(x), np.array(W1), np.array(W2)

    if mode == "post":
        y = x
        y = conv(y, W1)
        y = batch_norm(y, gamma1, beta1)
        y = relu(y)
        y = conv(y, W2)
        y = batch_norm(y, gamma2, beta2)
        y = x + y
        y = relu(y)
        
    if mode == "pre":
        y = x
        y = batch_norm(x, gamma1, beta1)
        y = relu(y)
        y = conv(y, W1)
        y = batch_norm(y, gamma2, beta2)
        y = relu(y)
        y = conv(y, W2)
        y = x + y
    
    y = np.round(y, 4)
    return {
        "output": y.tolist(),
        "mode": mode
    }
    