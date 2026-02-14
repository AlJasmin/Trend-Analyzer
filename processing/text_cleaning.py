from __future__ import annotations

import re
from typing import Optional

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
MD_SYMBOLS_RE = re.compile(r"[*_~]")
LINE_PREFIX_RE = re.compile(r"(?m)^[>\s#*\-\+]+\s*")
EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002700-\U000027bf"
    "\U0001f900-\U0001f9ff"
    "\U0001fa70-\U0001faff"
    "\U00002600-\U000026ff"
    "]+",
    flags=re.UNICODE,
)


def strip_markdown(text: str) -> str:
    text = CODE_BLOCK_RE.sub(" ", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = INLINE_CODE_RE.sub(r"\1", text)
    text = LINE_PREFIX_RE.sub("", text)
    text = MD_SYMBOLS_RE.sub("", text)
    return text


def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def remove_emojis(text: str) -> str:
    return EMOJI_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = remove_urls(text)
    cleaned = strip_markdown(cleaned)
    cleaned = remove_emojis(cleaned)
    return normalize_whitespace(cleaned)


__all__ = [
    "clean_text",
    "normalize_whitespace",
    "remove_emojis",
    "remove_urls",
    "strip_markdown",
]
