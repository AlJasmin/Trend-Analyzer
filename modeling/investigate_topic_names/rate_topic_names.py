from __future__ import annotations

import argparse
import json
import ast
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
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
        description="Rate topic_name/description fit per post."
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="Path to settings.yaml"
    )
    parser.add_argument(
        "--prompt",
        default=str(REPO_ROOT / "llm" / "prompts" / "check_topics.j2"),
        help="Prompt template path",
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Use compact prompt (post_id, matches, fit_score only)",
    )
    parser.add_argument(
        "--plot-output", default="plots/topic_name_ratings.png", help="Output plot path"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not store ratings in DB"
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=20000,
        help="Max input tokens per LLM call",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Max posts per LLM call (0 = no cap, only token budget)",
    )
    parser.add_argument(
        "--single-topic", default=None, help="Only rate posts from one topic_id"
    )
    parser.add_argument(
        "--debug-output", default=None, help="Optional directory for raw LLM responses"
    )
    parser.add_argument(
        "--max-retries", type=int, default=2, help="Retry splits when a batch fails"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of posts")
    parser.add_argument("--skip", type=int, default=0, help="Skip N posts")
    parser.add_argument(
        "--sample", type=int, default=0, help="Random sample size (overrides limit)"
    )
    return parser.parse_args()


def render_prompt(template_path: Path, context: Dict[str, str]) -> str:
    try:
        from jinja2 import Template
    except ImportError as exc:
        raise SystemExit(
            "jinja2 is required to render prompts. Install it first."
        ) from exc

    raw = template_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Prompt template is empty: {template_path}")

    return Template(raw).render(**context).strip()


def parse_response(raw: str) -> Optional[Any]:
    cleaned = raw.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            segment = part.strip()
            if segment.startswith("json"):
                segment = segment[4:].lstrip()
            if segment.startswith("{") or segment.startswith("["):
                cleaned = segment
                break
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        for start_char, end_char in (("[", "]"), ("{", "}")):
            start = cleaned.find(start_char)
            end = cleaned.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                snippet = cleaned[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    try:
                        return ast.literal_eval(snippet)
                    except (ValueError, SyntaxError):
                        continue
        try:
            return ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            pass
    return None


def estimate_tokens(text: str) -> int:
    return max(1, int(math.ceil(len(text) / 4)))


def build_entry(doc: Dict[str, Any]) -> Dict[str, str]:
    return {
        "post_id": str(doc.get("post_id") or ""),
        "topic_name": str(doc.get("topic_name") or ""),
        "topic_description": str(doc.get("topic_description") or ""),
        "topic_text": str(doc.get("topic_text") or ""),
    }


def dump_entries(entries: List[Dict[str, Any]]) -> str:
    return json.dumps(entries, ensure_ascii=True, separators=(",", ":"))


def pack_batches(
    docs: List[Dict[str, Any]],
    *,
    max_input_tokens: int,
    prompt_base_tokens: int,
    empty_payload_tokens: int,
    batch_size: int,
) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    batch_cap = batch_size if batch_size and batch_size > 0 else None

    for doc in docs:
        entry = build_entry(doc)
        entry_tokens = estimate_tokens(dump_entries([entry]))
        if prompt_base_tokens - empty_payload_tokens + entry_tokens > max_input_tokens:
            continue
        candidate_docs = current + [doc]
        candidate_entries = [build_entry(item) for item in candidate_docs]
        candidate_tokens = (
            prompt_base_tokens
            - empty_payload_tokens
            + estimate_tokens(dump_entries(candidate_entries))
        )
        if current and (
            candidate_tokens > max_input_tokens
            or (batch_cap and len(candidate_docs) > batch_cap)
        ):
            batches.append(current)
            current = [doc]
        else:
            current = candidate_docs

    if current:
        batches.append(current)

    return batches


def parse_batch_response(raw: str) -> List[Dict[str, Any]]:
    data = parse_response(raw)
    if data is None:
        return []
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return results
        items = data.get("items")
        if isinstance(items, list):
            return items
        payload = data.get("data")
        if isinstance(payload, list):
            return payload
        evaluations = data.get("evaluations")
        if isinstance(evaluations, list):
            return evaluations
        if "post_id" in data:
            return [data]
        return []
    if isinstance(data, list):
        return data
    return []


def iter_posts(
    collection,
    query: Dict[str, Any],
    *,
    limit: Optional[int],
    skip: int,
    sample: int,
) -> Iterable[Dict[str, Any]]:
    projection = {
        "_id": 0,
        "post_id": 1,
        "topic_text": 1,
        "topic_name": 1,
        "topic_description": 1,
    }

    if sample and sample > 0:
        pipeline = [
            {"$match": query},
            {"$sample": {"size": int(sample)}},
            {"$project": projection},
        ]
        for doc in collection.aggregate(pipeline):
            yield doc
        return

    cursor = collection.find(query, projection).skip(int(skip))
    if limit:
        cursor = cursor.limit(int(limit))
    for doc in cursor:
        yield doc


def main() -> None:
    args = parse_args()
    prompt_path = Path(args.prompt)
    if args.compact_output:
        prompt_path = REPO_ROOT / "llm" / "prompts" / "check_topics_compact.j2"
    output_path = Path(args.plot_output)
    debug_dir = Path(args.debug_output) if args.debug_output else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    store = connect_from_config(Path(args.config))
    try:
        query: Dict[str, Any] = {
            "topic_text": {"$nin": [None, ""]},
            "topic_name": {"$nin": [None, ""]},
            "topic_description": {"$nin": [None, ""]},
        }
        if args.single_topic:
            query["topic_id"] = args.single_topic
        docs = list(
            iter_posts(
                store.posts, query, limit=args.limit, skip=args.skip, sample=args.sample
            )
        )
        client = OpenRouterClient(config_path=Path(args.config))
        empty_payload = dump_entries([])
        prompt_base_tokens = estimate_tokens(
            render_prompt(prompt_path, {"POSTS_JSON": empty_payload})
        )
        empty_payload_tokens = estimate_tokens(empty_payload)
        scores: List[float] = []
        failures = 0

        if args.batch_size == 1:
            progress = tqdm(docs, total=len(docs), desc="Rate topics", unit="post")
            for index, doc in enumerate(progress, start=1):
                entry = build_entry(doc)
                payload = dump_entries([entry])
                prompt_tokens = (
                    prompt_base_tokens - empty_payload_tokens + estimate_tokens(payload)
                )
                if prompt_tokens > args.max_input_tokens:
                    failures += 1
                    continue
                prompt = render_prompt(prompt_path, {"POSTS_JSON": payload})
                response = client.generate_text(prompt)
                if not response:
                    if debug_dir:
                        (debug_dir / f"post_{index}_empty.txt").write_text(
                            "", encoding="utf-8"
                        )
                    failures += 1
                    continue
                if debug_dir:
                    (debug_dir / f"post_{index}.txt").write_text(
                        response, encoding="utf-8"
                    )

                results = parse_batch_response(response)
                if not results:
                    failures += 1
                    continue

                post_id = str(doc.get("post_id") or "")
                item = None
                for candidate in results:
                    if not isinstance(candidate, dict):
                        continue
                    if str(candidate.get("post_id") or "") == post_id:
                        item = candidate
                        break
                if item is None:
                    item = results[0] if results else None
                if not isinstance(item, dict):
                    failures += 1
                    continue

                score = item.get("fit_score")
                try:
                    rating = float(score)
                except (TypeError, ValueError):
                    failures += 1
                    continue

                scores.append(rating)
                if not args.dry_run:
                    store.posts.update_one(
                        {"post_id": post_id},
                        {"$set": {"topic_name_rating": rating}},
                    )
            progress.close()
        else:
            batches = pack_batches(
                docs,
                max_input_tokens=args.max_input_tokens,
                prompt_base_tokens=prompt_base_tokens,
                empty_payload_tokens=empty_payload_tokens,
                batch_size=args.batch_size,
            )
            queue: List[tuple[List[Dict[str, Any]], int]] = [
                (batch, 0) for batch in batches
            ]
            progress = tqdm(total=len(queue), desc="Rate topics", unit="batch")
            batch_index = 0
            while queue:
                batch, depth = queue.pop(0)
                batch_index += 1
                entries = [build_entry(doc) for doc in batch]
                payload = dump_entries(entries)
                prompt = render_prompt(prompt_path, {"POSTS_JSON": payload})
                response = client.generate_text(prompt)
                if not response:
                    if debug_dir:
                        (debug_dir / f"batch_{batch_index}_empty.txt").write_text(
                            "", encoding="utf-8"
                        )
                    if depth < args.max_retries and len(batch) > 1:
                        mid = max(1, len(batch) // 2)
                        queue.append((batch[:mid], depth + 1))
                        queue.append((batch[mid:], depth + 1))
                        progress.total += 2
                        progress.refresh()
                    else:
                        failures += len(batch)
                    progress.update(1)
                    continue
                if debug_dir:
                    (debug_dir / f"batch_{batch_index}.txt").write_text(
                        response, encoding="utf-8"
                    )

                results = parse_batch_response(response)
                if not results:
                    logger.warning(
                        "Empty parse for batch %s (first 200 chars): %s",
                        batch_index,
                        response[:200],
                    )
                    if depth < args.max_retries and len(batch) > 1:
                        mid = max(1, len(batch) // 2)
                        queue.append((batch[:mid], depth + 1))
                        queue.append((batch[mid:], depth + 1))
                        progress.total += 2
                        progress.refresh()
                    else:
                        failures += len(batch)
                    progress.update(1)
                    continue

                by_id: Dict[str, Dict[str, Any]] = {}
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    post_id = str(item.get("post_id") or "")
                    if post_id:
                        by_id[post_id] = item

                if not by_id and len(results) == len(batch):
                    for doc, item in zip(batch, results):
                        if not isinstance(item, dict):
                            failures += 1
                            continue
                        score = item.get("fit_score")
                        try:
                            rating = float(score)
                        except (TypeError, ValueError):
                            failures += 1
                            continue
                        scores.append(rating)
                        if not args.dry_run:
                            store.posts.update_one(
                                {"post_id": doc.get("post_id")},
                                {"$set": {"topic_name_rating": rating}},
                            )
                    progress.update(1)
                    continue

                missing = 0
                for doc in batch:
                    post_id = str(doc.get("post_id") or "")
                    item = by_id.get(post_id)
                    if not item:
                        missing += 1
                        continue
                    score = item.get("fit_score")
                    try:
                        rating = float(score)
                    except (TypeError, ValueError):
                        missing += 1
                        continue

                    scores.append(rating)
                    if not args.dry_run:
                        store.posts.update_one(
                            {"post_id": post_id},
                            {"$set": {"topic_name_rating": rating}},
                        )

                if missing and depth < args.max_retries and len(batch) > 1:
                    mid = max(1, len(batch) // 2)
                    queue.append((batch[:mid], depth + 1))
                    queue.append((batch[mid:], depth + 1))
                    progress.total += 2
                    progress.refresh()
                elif missing:
                    failures += missing

                progress.update(1)

            progress.close()
    finally:
        store.close()

    if not scores:
        logger.info("No ratings produced.")
        return

    avg_score = sum(scores) / len(scores)
    logger.info("Average topic_name rating: %.2f", avg_score)
    logger.info("Failed responses: %s", failures)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it first."
        ) from exc

    plt.figure(figsize=(9, 6))
    plt.hist(scores, bins=30, color="#4c72b0", alpha=0.8)
    plt.axvline(
        avg_score, color="#c0392b", linestyle="--", label=f"mean={avg_score:.1f}"
    )
    plt.xlabel("Topic name fit score")
    plt.ylabel("Post count")
    plt.title("Topic name rating distribution")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    logger.info("Plot saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
