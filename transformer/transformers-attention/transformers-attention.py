import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """

    d_k = K.shape[-1]

    # (query_len, d_q) @ (d_k, query_len) -> (query_len, query_len)
    attention_scores = Q @ K.transpose(-1, -2)
    scaled_attention_scores = attention_scores / (d_k ** 0.5)

    attention_weights = torch.softmax(scaled_attention_scores, dim=-1)
    
    # (query_len, query_len) @ (query_len, d_v) -> (query_len, d_v)
    return attention_weights @ V
    