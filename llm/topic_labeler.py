from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from db.store import connect_from_config  # noqa: E402
from llm.openrouter_client import OpenRouterClient  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label topics using stored topic_text values.")
    parser.add_argument("--config", default=None, help="Path to settings.yaml")
    parser.add_argument(
        "--prompt",
        default=str(REPO_ROOT / "llm" / "prompts" / "topic_label.j2"),
        help="Prompt template path",
    )
    parser.add_argument(
        "--posts-per-topic",
        type=int,
        default=6,
        help="Number of posts per topic_id",
    )
    parser.add_argument(
        "--min-topic-posts",
        type=int,
        default=1,
        help="Skip topic_ids with fewer posts",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Max chars per topic_text",
    )
    parser.add_argument(
        "--limit-topics",
        type=int,
        default=0,
        help="Limit number of topic_ids",
    )
    parser.add_argument("--topic-id", default=None, help="Only label a single topic_id")
    return parser.parse_args()


def render_prompt(template_path: Path, payload: str) -> str:
    try:
        from jinja2 import Template
    except ImportError as exc:
        raise SystemExit("jinja2 is required to render prompts. Install it first.") from exc

    raw = template_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Prompt template is empty: {template_path}")

    return Template(raw).render(text=payload).strip()


def normalize_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if max_chars > 0 and len(cleaned) > max_chars:
        return cleaned[:max_chars].rstrip() + "..."
    return cleaned


def build_payload(texts: List[str], max_chars: int) -> str:
    lines = []
    for idx, text in enumerate(texts):
        cleaned = normalize_text(text, max_chars)
        if cleaned:
            lines.append(f"{idx}. {cleaned}")
    return "\n\n".join(lines)


def parse_response(raw: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def fetch_topic_ids(posts, topic_id: Optional[str], limit: int) -> List[Any]:
    if topic_id is not None:
        return [topic_id]

    query = {"topic_id": {"$nin": [None, ""]}, "topic_text": {"$nin": [None, ""]}}
    topic_ids = posts.distinct("topic_id", query)
    topic_ids = sorted(topic_ids, key=lambda value: str(value))

    if limit and limit > 0:
        topic_ids = topic_ids[:limit]

    return topic_ids


def fetch_topic_texts(posts, topic_id: Any, limit: int) -> List[str]:
    query = {"topic_id": topic_id, "topic_text": {"$nin": [None, ""]}}
    cursor = posts.find(query, {"topic_text": 1, "score": 1, "created_utc": 1})
    cursor = cursor.sort([("score", -1), ("created_utc", -1)])
    if limit and limit > 0:
        cursor = cursor.limit(limit)
    texts = []
    for doc in cursor:
        text = doc.get("topic_text")
        if text:
            texts.append(text)
    return texts


def main() -> None:
    args = parse_args()
    config_path = Path(args.config) if args.config else None

    store = connect_from_config(config_path)
    try:
        topic_ids = fetch_topic_ids(store.posts, args.topic_id, args.limit_topics)
        if not topic_ids:
            logger.info("No topic_ids found. Nothing to label.")
            return

        client = OpenRouterClient(config_path=config_path)
        prompt_path = Path(args.prompt)

        for topic_id in topic_ids:
            texts = fetch_topic_texts(store.posts, topic_id, args.posts_per_topic)
            if len(texts) < args.min_topic_posts:
                logger.info("Skipping topic_id %s (only %s posts).", topic_id, len(texts))
                continue

            payload = build_payload(texts, args.max_chars)
            if not payload:
                logger.info("Skipping topic_id %s (empty payload).", topic_id)
                continue

            prompt = render_prompt(prompt_path, payload)
            response = client.generate_text(prompt)
            if not response:
                logger.warning("Empty response for topic_id %s.", topic_id)
                continue

            data = parse_response(response)
            if not data:
                logger.warning("Failed to parse JSON for topic_id %s: %s", topic_id, response[:200])
                continue

            topic_name = data.get("topic_name")
            topic_description = data.get("topic_description")
            if not topic_name or not topic_description:
                logger.warning("Missing topic_name/topic_description for topic_id %s.", topic_id)
                continue

            result = store.posts.update_many(
                {"topic_id": topic_id},
                {"$set": {"topic_name": topic_name, "topic_description": topic_description}},
            )
            logger.info("Updated %s posts for topic_id %s.", result.modified_count, topic_id)
    finally:
        store.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
