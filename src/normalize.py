import numpy as np

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    X: (N, D) oder (D,)
    """
    X = np.asarray(X)
    if X.ndim == 1:
        n = np.linalg.norm(X)
        return X / (n + eps)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)
