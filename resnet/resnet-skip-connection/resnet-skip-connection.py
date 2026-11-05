import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through residual Jacobians.
    """

    if gradients_F:
        gradients_F = np.array(gradients_F) # (n_layers, fan_out, fan_in)
        i = np.eye(gradients_F.shape[-1])
    
        from functools import reduce
        return x @ reduce(lambda l, r : l @ r, i + gradients_F)
    else:
        return x

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through plain Jacobians.
    """

    if gradients_F:
        gradients_F = np.array(gradients_F) # (n_layers, fan_out, fan_in)
    
        from functools import reduce
        return x @ reduce(lambda l, r : l @ r, gradients_F)
    else:
        return x