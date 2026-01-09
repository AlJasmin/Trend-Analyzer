from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to settings.yaml")
    parser.add_argument("--batch-size", type=int, default=1000, help="Bulk write batch size")
    parser.add_argument("--dry-run", action="store_true", help="Only show planned updates")
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


def build_update(row: Dict[str, Any]) -> Optional[UpdateOne]:
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

    return UpdateOne({"comment_id": comment_id}, {"$set": update}, upsert=False)


def flush_batch(collection, batch: list[UpdateOne], *, dry_run: bool) -> tuple[int, int]:
    if not batch:
        return 0, 0
    if dry_run:
        return len(batch), 0
    result = collection.bulk_write(batch, ordered=False)
    return result.matched_count, result.modified_count


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

    store = connect_from_config(Path(args.config))
    try:
        for row in iter_rows(input_path):
            total_rows += 1
            update = build_update(row)
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
    finally:
        store.close()

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()