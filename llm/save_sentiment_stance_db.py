from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pymongo import UpdateOne

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from db.store import CONFIG_PATH, connect_from_config  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Store stance/sentiment results from JSONL into MongoDB comments."
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "llm" / "stance_sentiment_results.jsonl"),
        help="Path to stance_sentiment_results.jsonl",
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="Path to settings.yaml"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Bulk write batch size"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only show planned updates"
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only update comments where sentiment_label and stance_label are missing/empty.",
    )
    parser.add_argument(
        "--report-missing",
        action="store_true",
        help="Write comment_ids from input that are missing in the DB",
    )
    parser.add_argument(
        "--missing-output",
        default="",
        help="Output path for missing comment_ids (default: llm/missing_comment_ids.txt)",
    )
    return parser.parse_args()


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line.")
                continue


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def build_update(row: Dict[str, Any], *, only_missing: bool) -> Optional[UpdateOne]:
    comment_id = str(row.get("comment_id") or "").strip()
    if not comment_id:
        return None

    update: Dict[str, Any] = {}
    sentiment = row.get("sentiment") or row.get("sentiment_label")
    stance = row.get("stance") or row.get("stance_label")
    if sentiment:
        update["sentiment_label"] = str(sentiment)
    if stance:
        update["stance_label"] = str(stance)

    confidence = parse_float(row.get("confidence"))
    if confidence is not None:
        update["llm_confidence"] = confidence

    rationale = row.get("rationale") or row.get("llm_comment_exp")
    if rationale is not None:
        update["llm_comment_exp"] = str(rationale)

    if not update:
        return None

    query: Dict[str, Any] = {"comment_id": comment_id}
    if only_missing:
        query["$and"] = [
            {
                "$or": [
                    {"sentiment_label": {"$exists": False}},
                    {"sentiment_label": None},
                    {"sentiment_label": ""},
                ]
            },
            {
                "$or": [
                    {"stance_label": {"$exists": False}},
                    {"stance_label": None},
                    {"stance_label": ""},
                ]
            },
        ]

    return UpdateOne(query, {"$set": update}, upsert=False)


def flush_batch(
    collection, batch: list[UpdateOne], *, dry_run: bool
) -> tuple[int, int]:
    if not batch:
        return 0, 0
    if dry_run:
        return len(batch), 0
    result = collection.bulk_write(batch, ordered=False)
    return result.matched_count, result.modified_count


def iter_chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def find_missing_comment_ids(
    collection,
    comment_ids: Iterable[str],
    *,
    chunk_size: int = 5000,
) -> List[str]:
    unique_ids = sorted({cid for cid in comment_ids if cid})
    if not unique_ids:
        return []

    found: set[str] = set()
    for chunk in iter_chunks(unique_ids, chunk_size):
        for doc in collection.find(
            {"comment_id": {"$in": chunk}}, {"comment_id": 1, "_id": 0}
        ):
            found.add(doc["comment_id"])

    return [cid for cid in unique_ids if cid not in found]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    batch_size = int(args.batch_size or 0)
    if batch_size <= 0:
        batch_size = 1000

    updates: list[UpdateOne] = []
    total_rows = 0
    total_updates = 0
    matched = 0
    modified = 0
    seen_comment_ids: List[str] = []

    store = connect_from_config(Path(args.config))
    missing_ids: Optional[List[str]] = None
    try:
        for row in iter_rows(input_path):
            total_rows += 1
            comment_id = str(row.get("comment_id") or "").strip()
            if comment_id:
                seen_comment_ids.append(comment_id)
            update = build_update(row, only_missing=args.only_missing)
            if update is None:
                continue
            updates.append(update)
            total_updates += 1

            if len(updates) >= batch_size:
                batch_matched, batch_modified = flush_batch(
                    store.comments,
                    updates,
                    dry_run=args.dry_run,
                )
                matched += batch_matched
                modified += batch_modified
                updates = []

        if updates:
            batch_matched, batch_modified = flush_batch(
                store.comments,
                updates,
                dry_run=args.dry_run,
            )
            matched += batch_matched
            modified += batch_modified
        if args.report_missing:
            missing_ids = find_missing_comment_ids(store.comments, seen_comment_ids)
    finally:
        store.close()

    if args.report_missing:
        missing_output = (
            Path(args.missing_output)
            if args.missing_output
            else REPO_ROOT / "llm" / "missing_comment_ids.txt"
        )
        missing_ids = missing_ids or []
        missing_output.parent.mkdir(parents=True, exist_ok=True)
        missing_output.write_text("\n".join(missing_ids), encoding="utf-8")
        logger.info(
            "Missing comment_ids: %s (written to %s).", len(missing_ids), missing_output
        )

    if args.dry_run:
        logger.info("Scanned %s rows, prepared %s updates.", total_rows, total_updates)
        return

    logger.info(
        "Scanned %s rows, updates=%s, matched=%s, modified=%s.",
        total_rows,
        total_updates,
        matched,
        modified,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
