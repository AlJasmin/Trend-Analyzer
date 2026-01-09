from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from db.store import CONFIG_PATH, connect_from_config  # noqa: E402
from reddit.reddit_cleaner import build_topic_text  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export post/comment pairs from MongoDB to JSONL.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to settings.yaml")
    parser.add_argument(
        "--output",
        default="local_LLM_test/sample.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--sample", type=int, default=200, help="Random sample size (0 = no sample)")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows when --sample is 0")
    parser.add_argument("--topic-id", default=None, help="Filter by topic_id")
    parser.add_argument("--subreddit", default=None, help="Filter by subreddit")
    parser.add_argument("--min-comment-chars", type=int, default=0, help="Minimum comment length")
    parser.add_argument("--max-post-chars", type=int, default=0, help="Trim post_text to N chars (0 = no trim)")
    parser.add_argument(
        "--max-comment-chars", type=int, default=0, help="Trim comment_text to N chars (0 = no trim)"
    )
    return parser.parse_args()


def shorten(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def build_pipeline(
    *,
    sample: int,
    limit: int,
    topic_id: Optional[str],
    subreddit: Optional[str],
) -> List[Dict[str, Any]]:
    match_comment = {
        "post_id": {"$exists": True, "$nin": [None, ""]},
        "$or": [
            {"comment_text_clean": {"$exists": True, "$ne": ""}},
            {"comment_text": {"$exists": True, "$ne": ""}},
            {"body_clean": {"$exists": True, "$ne": ""}},
            {"body": {"$exists": True, "$ne": ""}},
        ],
    }
    pipeline: List[Dict[str, Any]] = [
        {"$match": match_comment},
        {
            "$lookup": {
                "from": "posts",
                "localField": "post_id",
                "foreignField": "post_id",
                "as": "post",
            }
        },
        {"$unwind": "$post"},
        {
            "$match": {
                "$or": [
                    {"post.topic_text": {"$exists": True, "$ne": ""}},
                    {"post.title": {"$exists": True, "$ne": ""}},
                    {"post.selftext": {"$exists": True, "$ne": ""}},
                ]
            }
        },
    ]
    if topic_id:
        pipeline.append({"$match": {"post.topic_id": topic_id}})
    if subreddit:
        pipeline.append({"$match": {"post.subreddit": subreddit}})

    if sample and sample > 0:
        pipeline.append({"$sample": {"size": int(sample)}})
    elif limit and limit > 0:
        pipeline.append({"$limit": int(limit)})

    pipeline.append(
        {
            "$project": {
                "_id": 0,
                "comment_id": 1,
                "post_id": 1,
                "comment_text_clean": 1,
                "comment_text": 1,
                "body_clean": 1,
                "body": 1,
                "post": {
                    "topic_text": 1,
                    "title": 1,
                    "selftext": 1,
                },
            }
        }
    )
    return pipeline


def extract_pairs(
    docs: Iterable[Dict[str, Any]],
    *,
    min_comment_chars: int,
    max_post_chars: int,
    max_comment_chars: int,
) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for doc in docs:
        comment = (
            doc.get("comment_text_clean")
            or doc.get("comment_text")
            or doc.get("body_clean")
            or doc.get("body")
            or ""
        ).strip()
        if min_comment_chars and len(comment) < min_comment_chars:
            continue
        post_doc = doc.get("post") or {}
        post_text = (post_doc.get("topic_text") or "").strip()
        if not post_text:
            title = post_doc.get("title") or ""
            selftext = post_doc.get("selftext") or ""
            post_text = build_topic_text(title, selftext, bool(selftext))
        if not comment or not post_text:
            continue
        pairs.append(
            {
                "post_id": str(doc.get("post_id") or ""),
                "comment_id": str(doc.get("comment_id") or ""),
                "post_text": shorten(post_text, max_post_chars),
                "comment_text": shorten(comment, max_comment_chars),
            }
        )
    return pairs


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    store = connect_from_config(Path(args.config))
    try:
        pipeline = build_pipeline(
            sample=args.sample,
            limit=args.limit,
            topic_id=args.topic_id,
            subreddit=args.subreddit,
        )
        docs = list(store.comments.aggregate(pipeline))
    finally:
        store.close()

    pairs = extract_pairs(
        docs,
        min_comment_chars=args.min_comment_chars,
        max_post_chars=args.max_post_chars,
        max_comment_chars=args.max_comment_chars,
    )
    if not pairs:
        logger.info("No pairs found to export.")
        return

    output_path = Path(args.output)
    write_jsonl(output_path, pairs)
    logger.info("Wrote %s pairs to %s", len(pairs), output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
