import numpy as np

def get_alpha_bar(betas: list[float]) -> list[float]:
    """
    Returns the cumulative alpha-bar values rounded to six decimals.
    """
    
    betas = np.array(betas, dtype=np.float64)
    return np.cumprod(1 - betas).tolist()

def forward_diffusion(x_0: list, t: int, betas: list[float], epsilon: list) -> list:
    """
    Returns x_t with the same nested shape as x_0.
    """

    x_0, epsilon = np.array(x_0, dtype=np.float64), np.array(epsilon, dtype=np.float64)

    alpha_bar_t = get_alpha_bar(betas)[t-1]
    x_t = (alpha_bar_t ** 0.5) * x_0 + ((1 - alpha_bar_t) ** 0.5) * epsilon

    return x_t.tolist()