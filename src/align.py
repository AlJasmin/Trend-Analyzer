import numpy as np
from sklearn.decomposition import PCA
from joblib import dump, load

from .normalize import l2_normalize

class PCAAligner:
    """
    Aligns text and image embeddings to same dimension using separate PCA models.
    """

    def __init__(self, target_dim: int = 256, random_state: int = 0):
        self.target_dim = target_dim
        self.random_state = random_state
        self.pca_text = PCA(n_components=target_dim, random_state=random_state)
        self.pca_img = PCA(n_components=target_dim, random_state=random_state)
        self.is_fit = False

    def fit(self, text_embs: np.ndarray, img_embs: np.ndarray):
        text_norm = l2_normalize(text_embs)
        img_norm = l2_normalize(img_embs)

        self.pca_text.fit(text_norm)
        self.pca_img.fit(img_norm)
        self.is_fit = True
        return self

    def transform(self, text_embs: np.ndarray, img_embs: np.ndarray):
        if not self.is_fit:
            raise RuntimeError("PCAAligner is not fitted. Call fit() first.")

        text_norm = l2_normalize(text_embs)
        img_norm = l2_normalize(img_embs)

        text_a = self.pca_text.transform(text_norm)
        img_a = self.pca_img.transform(img_norm)

        # optional: normalize again after PCA (often helps clustering)
        text_a = l2_normalize(text_a)
        img_a = l2_normalize(img_a)
        return text_a, img_a

    def save(self, text_path: str, img_path: str):
        dump(self.pca_text, text_path)
        dump(self.pca_img, img_path)

    def load(self, text_path: str, img_path: str):
        self.pca_text = load(text_path)
        self.pca_img = load(img_path)
        self.is_fit = True
        return self
