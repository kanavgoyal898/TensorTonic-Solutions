import numpy as np

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """
    
    mu = np.mean(x, axis=-1, keepdims=True)
    sigma = np.std(x, axis=-1, keepdims=True)

    return gamma * (x - mu) / np.sqrt(sigma ** 2 + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns multi-head attention of Q, K, V, projecting them with W_q, W_k, W_v and the merged heads with W_o.
    """

    d_model = K.shape[-1]
    d_head = d_model // num_heads
    
    
    W_q = (
        W_q
            .reshape(*W_q.shape[:-2], W_q.shape[-2], num_heads, d_head)
            .swapaxes(-3, -2)
    )

    W_k = (
        W_k
            .reshape(*W_k.shape[:-2], W_k.shape[-2], num_heads, d_head)
            .swapaxes(-3, -2)
    )

    W_v = (
        W_v
            .reshape(*W_v.shape[:-2], W_v.shape[-2], num_heads, d_head)
            .swapaxes(-3, -2)
    )

    Q_proj = np.expand_dims(Q, -3) @ W_q
    K_proj = np.expand_dims(K, -3) @ W_k
    V_proj = np.expand_dims(V, -3) @ W_v

    attention_scores = Q_proj @ K_proj.swapaxes(-2, -1)
    scaled_attention_scores = attention_scores / d_head ** 0.5

    attention_weights = softmax(scaled_attention_scores, axis=-1)
    attention = attention_weights @ V_proj

    attention = attention.swapaxes(-3, -2)
    attention = attention.reshape(
        *attention.shape[:-3], attention.shape[-3], d_model
    )

    return attention @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    
    hidden_output = np.maximum(0, x @ W1 + b1)
    output = hidden_output @ W2 + b2
    
    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    
    z = layer_norm(
        x + multi_head_attention(
            x, x, x, W_q, W_k, W_v, W_o, num_heads
        ), gamma1, beta1
    )
    
    y = layer_norm(
        z + feed_forward(
            z, W1, b1, W2, b2
        ), gamma2, beta2
    )

    return y
