import numpy as np

def conv(x, W):
    x, W = np.array(x), np.array(W)
    return x @ W

def relu(x):
    return np.maximum(0, x)

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """

    y = conv(x, conv1)
    y = relu(y)

    x = y
    y = conv(y, W1_b1)
    y = relu(y)
    y = conv(y, W2_b1)
    y = x + y
    y = relu(y)

    x = y
    y = conv(y, W1_b2)
    y = relu(y)
    y = conv(y, W2_b2)
    y = conv(x, Ws_b2) + y
    y = relu(y)

    y = conv(y, fc)
    return np.round(y, 4).tolist()
    