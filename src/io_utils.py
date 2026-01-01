import numpy as np
from pathlib import Path

def load_npy(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(p, allow_pickle=False)

def save_npy(path: str, arr: np.ndarray):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr, allow_pickle=False)
