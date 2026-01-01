from __future__ import annotations

"""
LLaVA embedder for meme/screenshot images.

Computes normalized vision-tower embeddings for local files or URLs with a small LRU cache.
Defaults to llava-hf/llava-1.5-7b-hf.
"""

import io
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union, Dict, Any

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

logger = logging.getLogger(__name__)


def _move_inputs(inputs: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


class LlavaImageEmbedder:
    """Generate LLaVA vision-tower embeddings for local files or URLs with simple caching."""

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: Optional[str] = None,
        cache_size: int = 128,
        timeout: int = 10,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cache_size = cache_size
        self.timeout = timeout
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)

        logger.info("Loading LLaVA model %s on %s", model_name, self.device)

        # Processor knows how to preprocess images for this model
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # IMPORTANT FIX: Use the correct LLaVA model class (not AutoModelForCausalLM)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=None,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self._cache: "OrderedDict[str, list]" = OrderedDict()

    @staticmethod
    def _is_url(source: Union[str, Path]) -> bool:
        return str(source).lower().startswith(("http://", "https://"))

    @staticmethod
    def _resolve_path(path_like: Union[str, Path]) -> Path:
        path = Path(path_like).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image path not found: {path}")
        return path

    def _load_image(self, source: Union[str, Path]) -> Image.Image:
        if self._is_url(source):
            resp = requests.get(str(source), timeout=self.timeout)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")

        path = self._resolve_path(source)
        return Image.open(path).convert("RGB")

    def _cache_get(self, key: str) -> Optional[list]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: list) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _canonical_key(self, source: Union[str, Path]) -> str:
        if self._is_url(source):
            return str(source)
        return str(self._resolve_path(source))

    def _prepare_vision_inputs(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Returns a dict containing (at least) pixel_values as torch tensors.
        Works across processor variants.
        """
        # Most LLaVA processors expose image_processor
        if hasattr(self.processor, "image_processor"):
            out = self.processor.image_processor(images=image, return_tensors="pt")
            # out typically includes pixel_values
            return dict(out)

        # Fallback: try processor directly
        out = self.processor(images=image, return_tensors="pt")
        # keep only tensor fields
        return {k: v for k, v in out.items() if torch.is_tensor(v)}

    def _get_vision_module(self):
        """
        Find the vision tower / vision model in a robust way for different LLaVA HF implementations.
        """
        # Some models provide get_vision_tower()
        if hasattr(self.model, "get_vision_tower") and callable(getattr(self.model, "get_vision_tower")):
            vt = self.model.get_vision_tower()
            if vt is not None:
                return vt

        # Some expose vision_tower directly
        if hasattr(self.model, "vision_tower"):
            vt = self.model.vision_tower
            # Sometimes vision_tower is a list/ModuleList
            if isinstance(vt, (list, torch.nn.ModuleList)) and len(vt) > 0:
                return vt[0]
            return vt

        # Or vision_model
        if hasattr(self.model, "vision_model"):
            return self.model.vision_model

        raise AttributeError("LLaVA model missing vision_tower/vision_model/get_vision_tower().")

    def _forward_vision(self, vision_inputs: Dict[str, torch.Tensor]):
        """
        Forward pass through the vision tower and return its output.
        """
        vt = self._get_vision_module()

        # Common key is pixel_values. If processor returns something else, adjust here.
        # We'll pass everything we have; most vision towers accept pixel_values.
        if callable(vt):
            return vt(**vision_inputs)
        if hasattr(vt, "forward"):
            return vt(**vision_inputs)

        raise RuntimeError("Vision tower is not callable and has no forward().")

    def get_embedding(self, source: Union[str, Path]) -> list:
        """
        Compute a normalized LLaVA vision embedding for a local path or URL.

        Returns a Python list of floats to keep the interface JSON-friendly.
        """
        key = self._canonical_key(source)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        image = self._load_image(source)

        with torch.no_grad():
            vision_inputs = self._prepare_vision_inputs(image)
            vision_inputs = _move_inputs(vision_inputs, self.device, self.dtype)

            vision_out = self._forward_vision(vision_inputs)

            # Prefer pooler_output if present
            feats = getattr(vision_out, "pooler_output", None)

            # Fallback: average last_hidden_state tokens
            if feats is None and hasattr(vision_out, "last_hidden_state"):
                feats = vision_out.last_hidden_state.mean(dim=1)

            if feats is None:
                raise RuntimeError("Failed to extract vision features from LLaVA vision output.")

            # L2 normalize
            feats = feats / feats.norm(dim=-1, keepdim=True)

            vec = feats.squeeze(0).detach().cpu().float().numpy().tolist()

        self._cache_put(key, vec)
        return vec

    def clear_cache(self) -> None:
        self._cache.clear()


__all__ = ["LlavaImageEmbedder"]
