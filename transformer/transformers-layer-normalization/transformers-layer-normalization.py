import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """

    mu = np.mean(x, axis=-1, keepdims=True) # (..., d_model) -> (..., 1)
    sigma = np.std(x, axis=-1, keepdims=True) # (..., d_model) -> (..., 1)
    
    return gamma * (x - mu) / np.sqrt(sigma ** 2 + eps) + beta