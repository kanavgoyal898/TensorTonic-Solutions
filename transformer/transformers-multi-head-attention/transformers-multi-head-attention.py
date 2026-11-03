import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """

    d_model = K.shape[-1]
    d_head = d_model // num_heads 

    # (..., d_model, d_model) -> (..., num_heads, d_model, d_head)
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

    # (..., 1, seq_len, d_model) @ (..., num_heads, d_model, d_head)
    W_proj = np.expand_dims(Q, axis=-3) @ W_q # (..., num_heads, seq_len, d_head)
    K_proj = np.expand_dims(K, axis=-3) @ W_k # (..., num_heads, seq_len, d_head)
    V_proj = np.expand_dims(V, axis=-3) @ W_v # (..., num_heads, seq_len, d_head)

    # (..., num_heads, seq_len, seq_len)
    attention_scores = W_proj @ K_proj.swapaxes(-2, -1)
    scaled_attention_scores = attention_scores / (d_head ** 0.5)

    attention_weights = (
        np.exp(scaled_attention_scores)
            / np.sum(np.exp(scaled_attention_scores), axis=-1, keepdims=True)
    )

    # (..., num_heads, seq_len, seq_len) @ (..., num_heads, seq_len, d_head)
    attention = attention_weights @ V_proj # (..., num_heads, seq_len, d_head)
    attention = attention.swapaxes(-3, -2) # (..., seq_len, num_heads, d_head)
    
    attention = (
        attention
            .reshape(*attention.shape[:-2], -1) # (..., seq_len, d_model)
    )

    return attention @ W_o