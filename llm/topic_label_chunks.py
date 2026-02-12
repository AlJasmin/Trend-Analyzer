from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from db.store import CONFIG_PATH, connect_from_config  # noqa: E402
from reddit.reddit_cleaner import build_topic_text  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare topic-label chunks with centroid sampling and token estimates."
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="Path to settings.yaml"
    )
    parser.add_argument(
        "--prompt",
        default=str(REPO_ROOT / "llm" / "prompts" / "topic_label.j2"),
        help="Prompt template path",
    )
    parser.add_argument(
        "--ctfidf-csv",
        default="reports/ctfidf_topics.csv",
        help="Path to c-TF-IDF CSV",
    )
    parser.add_argument(
        "--output",
        default="reports/topic_label_chunks.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--posts-output",
        default="",
        help="Output CSV for per-post rows (default: derived from --output)",
    )
    parser.add_argument(
        "--max-input-tokens", type=int, default=20000, help="Token budget per chunk"
    )
    parser.add_argument(
        "--near-count", type=int, default=20, help="Posts nearest to centroid per topic"
    )
    parser.add_argument(
        "--far-count",
        type=int,
        default=5,
        help="Posts farthest from centroid per topic",
    )
    parser.add_argument(
        "--min-posts", type=int, default=1, help="Minimum posts per topic"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of posts to read"
    )
    parser.add_argument("--skip", type=int, default=0, help="Skip N posts")
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


def estimate_tokens(text: str) -> int:
    return max(1, int(math.ceil(len(text) / 4)))


def load_ctfidf_terms(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    terms_by_topic: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            topic_id = (row.get("topic_id") or "").strip()
            raw_terms = row.get("top_terms") or ""
            if not topic_id or not raw_terms:
                continue
            terms = [term for term in raw_terms.split("|") if term]
            terms_by_topic[topic_id] = terms
    return terms_by_topic


def iter_posts(
    collection,
    query: Dict[str, Any],
    *,
    limit: Optional[int],
    skip: int,
) -> Iterable[Dict[str, Any]]:
    projection = {
        "_id": 0,
        "post_id": 1,
        "topic_id": 1,
        "topic_text": 1,
        "title": 1,
        "selftext": 1,
        "center_distance": 1,
    }
    cursor = collection.find(query, projection).skip(int(skip))
    if limit:
        cursor = cursor.limit(int(limit))
    for doc in cursor:
        yield doc


def get_topic_text(doc: Dict[str, Any]) -> str:
    text = (doc.get("topic_text") or "").strip()
    if text:
        return text
    title = doc.get("title", "")
    selftext = doc.get("selftext", "")
    return build_topic_text(title, selftext, bool(selftext))


def collect_posts(
    collection,
    *,
    min_posts: int,
    limit: Optional[int],
    skip: int,
) -> Dict[str, List[Dict[str, Any]]]:
    query: Dict[str, Any] = {"topic_id": {"$exists": True, "$ne": None}}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for doc in iter_posts(collection, query, limit=limit, skip=skip):
        topic_id = str(doc.get("topic_id") or "")
        if not topic_id:
            continue
        if topic_id == "noise":
            continue
        distance = doc.get("center_distance")
        if distance is None:
            continue
        text = get_topic_text(doc)
        if not text:
            continue
        grouped.setdefault(topic_id, []).append(
            {
                "post_id": str(doc.get("post_id") or ""),
                "distance": float(distance),
                "text": text,
            }
        )

    if min_posts <= 1:
        return grouped

    filtered: Dict[str, List[Dict[str, Any]]] = {}
    for topic_id, posts in grouped.items():
        if len(posts) >= min_posts:
            filtered[topic_id] = posts
    return filtered


def select_posts(
    posts: List[Dict[str, Any]],
    *,
    near_count: int,
    far_count: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    if not posts:
        return []
    ordered = sorted(posts, key=lambda item: item["distance"])
    total = len(ordered)
    near_count = max(0, min(near_count, total))
    far_count = max(0, min(far_count, total))

    selected: List[Tuple[str, Dict[str, Any]]] = []
    near_set = set(range(near_count))
    far_set = set(range(max(0, total - far_count), total))

    if total <= near_count + far_count:
        for idx, item in enumerate(ordered):
            tag = "NEAR" if idx in near_set else "FAR"
            selected.append((tag, item))
        return selected

    for idx in sorted(near_set):
        selected.append(("NEAR", ordered[idx]))
    for idx in sorted(far_set):
        if idx in near_set:
            continue
        selected.append(("FAR", ordered[idx]))
    return selected


def build_entry(tag: str, item: Dict[str, Any], index: int) -> str:
    text = item["text"].strip()
    return f"[{index}] tag={tag} post_id={item['post_id']} distance={item['distance']:.6f}\n{text}"


def build_payload(
    header: str,
    entries: List[Tuple[str, Dict[str, Any]]],
) -> str:
    sections: List[str] = []
    if header:
        sections.append(header)
    post_lines = [
        build_entry(tag, item, idx) for idx, (tag, item) in enumerate(entries)
    ]
    sections.append(
        "POSTS (tagged NEAR/FAR; lower distance = more central):\n"
        + "\n\n".join(post_lines)
    )
    return "\n\n".join(sections).strip()


def build_chunks(
    entries: List[Tuple[str, Dict[str, Any]]],
    *,
    ctfidf_terms: List[str],
    prompt_path: Path,
    max_tokens: int,
) -> List[Tuple[List[Tuple[str, Dict[str, Any]]], int]]:
    chunks: List[Tuple[List[Tuple[str, Dict[str, Any]]], int]] = []
    header = ""
    if ctfidf_terms:
        header = "CTFIDF_TERMS: " + ", ".join(ctfidf_terms)

    current: List[Tuple[str, Dict[str, Any]]] = []
    current_tokens = 0
    for entry in entries:
        candidate = current + [entry]
        payload = build_payload(header, candidate)
        tokens = estimate_tokens(render_prompt(prompt_path, payload))

        if tokens > max_tokens:
            if current:
                chunks.append((current, current_tokens))
                current = []
                current_tokens = 0

                payload = build_payload(header, [entry])
                tokens = estimate_tokens(render_prompt(prompt_path, payload))
                if tokens > max_tokens:
                    logger.warning(
                        "Skipping post_id %s; single entry exceeds token budget.",
                        entry[1].get("post_id"),
                    )
                    continue
                current = [entry]
                current_tokens = tokens
            else:
                logger.warning(
                    "Skipping post_id %s; single entry exceeds token budget.",
                    entry[1].get("post_id"),
                )
            continue

        current = candidate
        current_tokens = tokens

    if current:
        chunks.append((current, current_tokens))

    return chunks


def write_chunks_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "topic_id",
                "chunk_id",
                "post_count",
                "near_count",
                "far_count",
                "token_estimate",
                "cluster_token_sum",
                "ctfidf_terms",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_posts_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "topic_id",
                "chunk_id",
                "post_index",
                "tag",
                "post_id",
                "distance",
                "text",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def derive_posts_output(chunks_output: Path) -> Path:
    stem = chunks_output.stem
    if "chunks" in stem:
        new_stem = stem.replace("chunks", "posts")
    elif "chunk" in stem:
        new_stem = stem.replace("chunk", "posts")
    else:
        new_stem = stem + "_posts"
    return chunks_output.with_name(new_stem + chunks_output.suffix)


def main() -> None:
    args = parse_args()
    prompt_path = Path(args.prompt)
    ctfidf_terms = load_ctfidf_terms(Path(args.ctfidf_csv))

    store = connect_from_config(Path(args.config))
    try:
        grouped = collect_posts(
            store.posts,
            min_posts=args.min_posts,
            limit=args.limit,
            skip=args.skip,
        )
    finally:
        store.close()

    if not grouped:
        logger.info("No topics found to build chunks.")
        return

    chunk_rows: List[Dict[str, Any]] = []
    post_rows: List[Dict[str, Any]] = []
    token_sums: Dict[str, int] = {}
    for topic_id, posts in sorted(grouped.items()):
        selected = select_posts(
            posts,
            near_count=args.near_count,
            far_count=args.far_count,
        )
        if not selected:
            continue
        terms = ctfidf_terms.get(topic_id, [])
        chunks = build_chunks(
            selected,
            ctfidf_terms=terms,
            prompt_path=prompt_path,
            max_tokens=args.max_input_tokens,
        )
        ctfidf_str = "|".join(terms)
        for chunk_idx, (entries, tokens) in enumerate(chunks, start=1):
            near_count = sum(1 for tag, _ in entries if tag == "NEAR")
            far_count = sum(1 for tag, _ in entries if tag == "FAR")
            token_sums[topic_id] = token_sums.get(topic_id, 0) + int(tokens)
            chunk_rows.append(
                {
                    "topic_id": topic_id,
                    "chunk_id": chunk_idx,
                    "post_count": len(entries),
                    "near_count": near_count,
                    "far_count": far_count,
                    "token_estimate": tokens,
                    "cluster_token_sum": 0,
                    "ctfidf_terms": ctfidf_str,
                }
            )
            for post_index, (tag, item) in enumerate(entries):
                post_rows.append(
                    {
                        "topic_id": topic_id,
                        "chunk_id": chunk_idx,
                        "post_index": post_index,
                        "tag": tag,
                        "post_id": item["post_id"],
                        "distance": f"{item['distance']:.6f}",
                        "text": item["text"],
                    }
                )

    for row in chunk_rows:
        row["cluster_token_sum"] = token_sums.get(str(row["topic_id"]), 0)

    chunks_output = Path(args.output)
    posts_output = (
        Path(args.posts_output)
        if args.posts_output
        else derive_posts_output(chunks_output)
    )
    write_chunks_csv(chunks_output, chunk_rows)
    write_posts_csv(posts_output, post_rows)
    logger.info("Wrote %s chunks to %s", len(chunk_rows), chunks_output)
    logger.info("Wrote %s posts to %s", len(post_rows), posts_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
