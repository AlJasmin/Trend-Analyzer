from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from db.store import CONFIG_PATH, connect_from_config  # noqa: E402
from llm.openrouter_client import OpenRouterClient  # noqa: E402

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress

    class _NullTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self._iterable = iterable or []

        def __iter__(self):
            return iter(self._iterable)

        def update(self, n=1):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, **kwargs):
        return _NullTqdm(iterable, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label clusters with LLM using sampled posts."
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="Path to settings.yaml"
    )
    parser.add_argument(
        "--prompt",
        default=str(REPO_ROOT / "llm" / "prompts" / "topic_label.j2"),
        help="Prompt template path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Max chars per topic_text (0 = no trim)",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=20000,
        help="Max input tokens per LLM call",
    )
    parser.add_argument(
        "--min-words", type=int, default=20, help="Minimum words per post"
    )
    parser.add_argument(
        "--max-words-trim-percent",
        type=float,
        default=5.0,
        help="Drop longest X%% of posts per topic based on word count",
    )
    parser.add_argument(
        "--dominant-threshold",
        type=float,
        default=0.25,
        help="Dominant ratio threshold",
    )
    parser.add_argument(
        "--dominant-sample-ratio",
        type=float,
        default=0.1,
        help="Sample ratio for dominant",
    )
    parser.add_argument(
        "--dominant-cap", type=int, default=80, help="Max samples for dominant cluster"
    )
    parser.add_argument(
        "--non-dominant-sample",
        type=int,
        default=50,
        help="Sample size for non-dominant",
    )
    parser.add_argument(
        "--topic-id", default=None, help="Only process a single topic_id"
    )
    parser.add_argument(
        "--limit-topics", type=int, default=0, help="Limit number of topics processed"
    )
    return parser.parse_args()


def render_prompt(template_path: Path, payload: str) -> str:
    try:
        from jinja2 import Template
    except ImportError as exc:
        raise SystemExit(
            "jinja2 is required to render prompts. Install it first."
        ) from exc

    raw = template_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Prompt template is empty: {template_path}")

    return Template(raw).render(text=payload).strip()


def normalize_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if max_chars and max_chars > 0 and len(cleaned) > max_chars:
        return cleaned[:max_chars].rstrip() + "..."
    return cleaned


def build_payload(texts: List[str], max_chars: int) -> str:
    lines = []
    for idx, text in enumerate(texts):
        cleaned = normalize_text(text, max_chars)
        if cleaned:
            lines.append(f"{idx}. {cleaned}")
    return "\n\n".join(lines)


def estimate_tokens(text: str) -> int:
    return max(1, int(math.ceil(len(text) / 4)))


def annotate_tokens(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for doc in docs:
        text = doc.get("topic_text") or ""
        doc["_token_est"] = estimate_tokens(text)
    return docs


def filter_too_large(
    docs: List[Dict[str, Any]], max_input_tokens: int
) -> Tuple[List[Dict[str, Any]], int]:
    filtered: List[Dict[str, Any]] = []
    skipped = 0
    for doc in docs:
        tokens = int(doc.get("_token_est") or 0)
        if tokens > max_input_tokens:
            skipped += 1
            continue
        filtered.append(doc)
    return filtered, skipped


def pack_by_token_budget(
    docs: List[Dict[str, Any]],
    *,
    max_input_tokens: int,
    prompt_base_tokens: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    packed: List[Dict[str, Any]] = []
    dropped_for_budget = 0
    total_tokens = prompt_base_tokens

    for doc in docs:
        tokens = int(doc.get("_token_est") or 0)
        if total_tokens + tokens > max_input_tokens:
            dropped_for_budget += 1
            continue
        packed.append(doc)
        total_tokens += tokens

    return packed, dropped_for_budget, total_tokens


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


def is_noise_topic(topic_id: Any) -> bool:
    return str(topic_id) == "noise"


def is_noise_cluster(topic_id: Any) -> bool:
    return str(topic_id).startswith("noise_")


def fetch_topic_counts(posts) -> Dict[Any, int]:
    pipeline = [
        {
            "$match": {
                "topic_id": {"$nin": [None, ""]},
                "topic_text": {"$nin": [None, ""]},
            }
        },
        {"$group": {"_id": "$topic_id", "count": {"$sum": 1}}},
    ]
    counts: Dict[Any, int] = {}
    for doc in posts.aggregate(pipeline):
        counts[doc["_id"]] = int(doc.get("count") or 0)
    return counts


def word_count(text: str) -> int:
    return len(text.split())


def percentile(values: List[int], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values_sorted[int(k)])
    lower = values_sorted[f]
    upper = values_sorted[c]
    return float(lower + (upper - lower) * (k - f))


def fetch_posts(
    posts, topic_id: Any, min_words: int, max_words_trim_percent: float
) -> List[Dict[str, Any]]:
    query = {"topic_id": topic_id, "topic_text": {"$nin": [None, ""]}}
    projection = {"_id": 0, "post_id": 1, "topic_text": 1, "score": 1, "created_utc": 1}
    docs = list(posts.find(query, projection))
    if min_words <= 0:
        filtered = docs
    else:
        filtered = [
            doc for doc in docs if word_count(doc.get("topic_text") or "") >= min_words
        ]

    trim_percent = min(max(max_words_trim_percent, 0.0), 99.0)
    if trim_percent <= 0 or not filtered:
        return filtered

    counts = [word_count(doc.get("topic_text") or "") for doc in filtered]
    cutoff_pct = 100.0 - trim_percent
    cutoff = percentile(counts, cutoff_pct)
    trimmed = [
        doc for doc in filtered if word_count(doc.get("topic_text") or "") <= cutoff
    ]
    return trimmed or filtered


def take_top_score(
    docs: List[Dict[str, Any]], n: int, taken: set[str]
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for doc in sorted(docs, key=lambda d: int(d.get("score") or 0), reverse=True):
        if len(selected) >= n:
            break
        post_id = str(doc.get("post_id"))
        if post_id in taken:
            continue
        selected.append(doc)
        taken.add(post_id)
    return selected


def take_time_split(
    docs: List[Dict[str, Any]], n: int, taken: set[str]
) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    ordered = sorted(docs, key=lambda d: int(d.get("created_utc") or 0))
    early_n = n // 2
    late_n = n - early_n
    selected: List[Dict[str, Any]] = []
    for doc in ordered:
        if len(selected) >= early_n:
            break
        post_id = str(doc.get("post_id"))
        if post_id in taken:
            continue
        selected.append(doc)
        taken.add(post_id)
    for doc in reversed(ordered):
        if len(selected) >= n:
            break
        post_id = str(doc.get("post_id"))
        if post_id in taken:
            continue
        selected.append(doc)
        taken.add(post_id)
    return selected


def take_random(
    docs: List[Dict[str, Any]], n: int, taken: set[str], rng: random.Random
) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    remaining = [doc for doc in docs if str(doc.get("post_id")) not in taken]
    rng.shuffle(remaining)
    selected = remaining[:n]
    for doc in selected:
        taken.add(str(doc.get("post_id")))
    return selected


def sample_dominant(
    docs: List[Dict[str, Any]], sample_size: int, rng: random.Random
) -> List[Dict[str, Any]]:
    if sample_size >= len(docs):
        return docs
    base = sample_size // 3
    remainder = sample_size % 3
    score_n = base + (1 if remainder > 0 else 0)
    rand_n = base + (1 if remainder > 1 else 0)
    time_n = base

    taken: set[str] = set()
    selected: List[Dict[str, Any]] = []
    selected += take_top_score(docs, score_n, taken)
    selected += take_time_split(docs, time_n, taken)
    selected += take_random(docs, rand_n, taken, rng)
    if len(selected) < sample_size:
        selected += take_random(docs, sample_size - len(selected), taken, rng)
    return selected


def sample_non_dominant(
    docs: List[Dict[str, Any]], sample_size: int, rng: random.Random
) -> List[Dict[str, Any]]:
    if sample_size >= len(docs):
        return docs
    score_n = sample_size // 2
    rand_n = sample_size - score_n
    taken: set[str] = set()
    selected: List[Dict[str, Any]] = []
    selected += take_top_score(docs, score_n, taken)
    selected += take_random(docs, rand_n, taken, rng)
    if len(selected) < sample_size:
        selected += take_random(docs, sample_size - len(selected), taken, rng)
    return selected


def label_topic(
    client: OpenRouterClient,
    prompt_path: Path,
    topic_id: Any,
    docs: List[Dict[str, Any]],
    max_chars: int,
    max_input_tokens: int,
    prompt_base_tokens: int,
) -> Optional[Dict[str, Any]]:
    docs = annotate_tokens(docs)
    docs, skipped_too_large = filter_too_large(docs, max_input_tokens)
    if not docs:
        logger.warning(
            "Skipping topic_id %s (all posts exceed token budget, too_large=%s).",
            topic_id,
            skipped_too_large,
        )
        return None

    attempt = 0
    max_attempts = 3
    dropped_for_budget = 0
    packed: List[Dict[str, Any]] = []
    data: Optional[Dict[str, Any]] = None
    while attempt < max_attempts:
        packed, dropped_for_budget, _ = pack_by_token_budget(
            docs,
            max_input_tokens=max_input_tokens,
            prompt_base_tokens=prompt_base_tokens,
        )
        if not packed:
            logger.warning(
                "Skipping topic_id %s (no posts fit token budget, too_large=%s, dropped=%s).",
                topic_id,
                skipped_too_large,
                dropped_for_budget,
            )
            return None

        texts = [doc.get("topic_text") or "" for doc in packed]
        payload = build_payload(texts, max_chars)
        if not payload:
            logger.warning("Skipping topic_id %s (empty payload).", topic_id)
            return None

        prompt = render_prompt(prompt_path, payload)
        response = client.generate_text(prompt)
        if not response:
            logger.warning(
                "Empty response for topic_id %s (attempt %s/%s).",
                topic_id,
                attempt + 1,
                max_attempts,
            )
        else:
            data = parse_response(response)
            if data:
                topic_name = data.get("topic_name")
                topic_description = data.get("topic_description")
                if topic_name and topic_description:
                    break
                logger.warning(
                    "Missing topic_name/topic_description for topic_id %s (attempt %s/%s).",
                    topic_id,
                    attempt + 1,
                    max_attempts,
                )
            else:
                logger.warning(
                    "Failed to parse JSON for topic_id %s (attempt %s/%s).",
                    topic_id,
                    attempt + 1,
                    max_attempts,
                )

        attempt += 1
        if len(packed) <= 1:
            data = None
            break
        docs = packed[: max(1, len(packed) // 2)]

    if not data:
        return None

    topic_name = data.get("topic_name")
    topic_description = data.get("topic_description")
    if not topic_name or not topic_description:
        return None

    confidence = data.get("confidence")
    return {
        "topic_name": str(topic_name),
        "topic_description": str(topic_description),
        "confidence": confidence,
        "skipped_too_large": skipped_too_large,
        "dropped_for_budget": dropped_for_budget,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    prompt_path = Path(args.prompt)

    store = connect_from_config(Path(args.config))
    try:
        counts = fetch_topic_counts(store.posts)
        if not counts:
            logger.info("No topic_id values found to label.")
            return

        normal_topics = [
            tid
            for tid in counts
            if not is_noise_topic(tid) and not is_noise_cluster(tid)
        ]
        noise_topics = [tid for tid in counts if is_noise_cluster(tid)]

        if args.topic_id:
            if args.topic_id in counts:
                normal_topics = (
                    [args.topic_id] if args.topic_id in normal_topics else []
                )
                noise_topics = [args.topic_id] if args.topic_id in noise_topics else []
            else:
                logger.info("topic_id %s not found.", args.topic_id)
                return

        if args.limit_topics and args.limit_topics > 0:
            normal_topics = normal_topics[: args.limit_topics]

        total_non_noise = sum(counts[tid] for tid in normal_topics)
        if total_non_noise == 0 and not noise_topics:
            logger.info("No non-noise clusters found.")
            return

        client = OpenRouterClient(config_path=Path(args.config))
        prompt_base_tokens = estimate_tokens(render_prompt(prompt_path, ""))
        total_too_large = 0
        total_dropped = 0
        total_labeled = 0

        for topic_id in tqdm(normal_topics, desc="Label topics", unit="topic"):
            count = counts[topic_id]
            docs = fetch_posts(
                store.posts, topic_id, args.min_words, args.max_words_trim_percent
            )
            if not docs:
                continue

            if count < args.non_dominant_sample:
                sample = docs
            else:
                dominant = count > (args.dominant_threshold * total_non_noise)
                if dominant:
                    sample_size = int(math.ceil(count * args.dominant_sample_ratio))
                    sample_size = min(sample_size, args.dominant_cap)
                    sample = sample_dominant(docs, sample_size, rng)
                else:
                    sample = sample_non_dominant(docs, args.non_dominant_sample, rng)

            result = label_topic(
                client,
                prompt_path,
                topic_id,
                sample,
                args.max_chars,
                args.max_input_tokens,
                prompt_base_tokens,
            )
            if not result:
                continue
            total_too_large += int(result.get("skipped_too_large") or 0)
            total_dropped += int(result.get("dropped_for_budget") or 0)
            store.posts.update_many(
                {"topic_id": topic_id},
                {
                    "$set": {
                        "topic_name": result["topic_name"],
                        "topic_description": result["topic_description"],
                        "confidence": result.get("confidence"),
                    }
                },
            )
            total_labeled += 1
            logger.info("Stored label for %s.", topic_id)

        if noise_topics:
            min_noise = min(counts[tid] for tid in noise_topics)
            for topic_id in tqdm(noise_topics, desc="Label noise topics", unit="topic"):
                docs = fetch_posts(
                    store.posts, topic_id, args.min_words, args.max_words_trim_percent
                )
                if not docs:
                    continue
                sample = sample_non_dominant(docs, min_noise, rng)
                result = label_topic(
                    client,
                    prompt_path,
                    topic_id,
                    sample,
                    args.max_chars,
                    args.max_input_tokens,
                    prompt_base_tokens,
                )
                if not result:
                    continue
                total_too_large += int(result.get("skipped_too_large") or 0)
                total_dropped += int(result.get("dropped_for_budget") or 0)
                store.posts.update_many(
                    {"topic_id": topic_id},
                    {
                        "$set": {
                            "topic_name": result["topic_name"],
                            "topic_description": result["topic_description"],
                            "confidence": result.get("confidence"),
                        }
                    },
                )
                total_labeled += 1
                logger.info("Stored label for %s.", topic_id)

        logger.info("Labeled %s topics.", total_labeled)
        logger.info("Skipped %s posts (too large).", total_too_large)
        logger.info("Dropped %s posts (token budget).", total_dropped)
    finally:
        store.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
