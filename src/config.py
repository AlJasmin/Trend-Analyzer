from dataclasses import dataclass

@dataclass
class FusionConfig:
    target_dim: int = 256          # PCA-Zieldimension
    method: str = "concat"         # "concat" oder "wsum"
    alpha: float = 0.7             # nur für "wsum"
    normalize_after: bool = True   # nach Fusion nochmal L2-normalisieren
    use_has_image_flag: bool = False  # optional: has_image als extra Feature anhängen