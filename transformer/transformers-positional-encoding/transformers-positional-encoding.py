import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """

    i = np.arange(0, d_model, 2)   # (d_model / 2, )
    
    freq = 10000 ** (i / d_model)  # (d_model / 2, )
    freq = freq[np.newaxis, :]     # (1, d_model / 2)

    pos = np.arange(seq_length)    # (seq_length, )
    pos = pos[:, np.newaxis]       # (seq_length, 1)
    
    sin_pe = np.sin(pos / freq)    # (seq_length, d_model / 2)
    cos_pe = np.cos(pos / freq)    # (seq_length, d_model / 2)

    pe = np.zeros((seq_length, d_model))  # (seq_length, d_model)
    pe[:, 0::2] = sin_pe
    pe[:, 1::2] = cos_pe

    return pe
    