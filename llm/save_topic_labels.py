from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from db.store import CONFIG_PATH, connect_from_config  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Store topic labels from topic_label_results.csv into MongoDB."
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="Path to settings.yaml"
    )
    parser.add_argument(
        "--results-csv",
        default="reports/topic_label_results.csv",
        help="Path to topic_label_results.csv",
    )
    parser.add_argument(
        "--topic-id", default=None, help="Only update a single topic_id"
    )
    parser.add_argument(
        "--topic-ids",
        nargs="+",
        default=None,
        help="Only update the specified topic_ids (space-separated)",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow empty topic_name/topic_description updates",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show planned updates only"
    )
    return parser.parse_args()


def iter_rows(path: Path) -> Iterable[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            yield row


def parse_confidence(value: str) -> Optional[float]:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def pick_best_per_topic(
    rows: Iterable[Dict[str, str]],
    *,
    topic_filters: Optional[set[str]],
    allow_empty: bool,
) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        topic_id = (row.get("topic_id") or "").strip()
        if not topic_id:
            continue
        if topic_filters is not None and topic_id not in topic_filters:
            continue

        status = (row.get("status") or "").strip().lower()
        if status and status != "ok":
            continue

        topic_name = (row.get("topic_name") or "").strip()
        topic_description = (row.get("topic_description") or "").strip()
        if not allow_empty and (not topic_name or not topic_description):
            continue

        confidence = parse_confidence(row.get("confidence") or "")
        score = confidence if confidence is not None else -1.0

        current = best.get(topic_id)
        if current is None or score > current["score"]:
            best[topic_id] = {
                "topic_name": topic_name,
                "topic_description": topic_description,
                "confidence": confidence,
                "score": score,
            }
    return best


def build_update(data: Dict[str, Any]) -> Dict[str, Any]:
    update = {
        "topic_name": data.get("topic_name") or "",
        "topic_description": data.get("topic_description") or "",
    }
    if data.get("confidence") is not None:
        update["confidence"] = float(data["confidence"])
    return update


def main() -> None:
    args = parse_args()
    topic_filters: Optional[set[str]] = None
    if args.topic_id or args.topic_ids:
        topic_filters = set()
        if args.topic_id:
            topic_filters.add(args.topic_id)
        if args.topic_ids:
            topic_filters.update(args.topic_ids)

    best = pick_best_per_topic(
        iter_rows(Path(args.results_csv)),
        topic_filters=topic_filters,
        allow_empty=args.allow_empty,
    )

    if not best:
        logger.info("No valid topic labels found to store.")
        return

    if args.dry_run:
        for topic_id, data in sorted(best.items()):
            update = build_update(data)
            logger.info("Would update %s: %s", topic_id, update)
        return

    store = connect_from_config(Path(args.config))
    try:
        for topic_id, data in sorted(best.items()):
            update = build_update(data)
            result = store.posts.update_many({"topic_id": topic_id}, {"$set": update})
            logger.info(
                "Updated %s posts for %s (matched=%s).",
                result.modified_count,
                topic_id,
                result.matched_count,
            )
    finally:
        store.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
