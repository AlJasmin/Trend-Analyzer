from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def load_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import ruamel.yaml as ry

        with path.open("r", encoding="utf-8") as fp:
            return ry.YAML().load(fp) or {}
    except Exception:
        import yaml

        with path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}


def _get_openrouter_config(settings: Mapping[str, Any]) -> Dict[str, Any]:
    return settings.get("openrouter") or {}


class OpenRouterClient:
    def __init__(
        self,
        *,
        config_path: Optional[Path] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        settings = load_settings(config_path or CONFIG_PATH)
        cfg = _get_openrouter_config(settings)

        key = api_key or os.getenv("OPENROUTER_API_KEY") or cfg.get("api_key")
        if not key:
            raise ValueError("OpenRouter API key missing (OPENROUTER_API_KEY or openrouter.api_key).")

        self.model = model or cfg.get("model") or "openai/gpt-4o-mini"
        self.temperature = temperature if temperature is not None else float(cfg.get("temperature", 0.3))
        self.max_tokens = max_tokens if max_tokens is not None else int(cfg.get("max_tokens", 512))
        self.timeout = timeout if timeout is not None else float(cfg.get("timeout", 60))
        self.max_retries = max_retries if max_retries is not None else int(cfg.get("max_retries", 2))

        headers: Dict[str, str] = {}
        referer = cfg.get("referer") or os.getenv("OPENROUTER_HTTP_REFERER")
        app_name = cfg.get("app_name") or os.getenv("OPENROUTER_APP_NAME")
        if referer:
            headers["HTTP-Referer"] = str(referer)
        if app_name:
            headers["X-Title"] = str(app_name)

        client_kwargs: Dict[str, Any] = {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if headers:
            client_kwargs["default_headers"] = headers

        self.client = OpenAI(**client_kwargs)

        logger.info("OpenRouter client initialized with model=%s", self.model)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = self.chat_raw(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return self._extract_content(response)

    def chat_raw(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        return self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )

    @staticmethod
    def _extract_content(response) -> str:
        if not response or not getattr(response, "choices", None):
            return ""

        message = response.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                else:
                    parts.append(str(part))
            return "".join(parts).strip()
        if isinstance(content, str):
            return content.strip()
        return ""

    def generate_text(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)
