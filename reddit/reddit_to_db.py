from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from reddit.reddit_scraper import run_pipeline
    from db.store import connect_from_config
except ImportError:
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from reddit.reddit_scraper import run_pipeline
    from db.store import connect_from_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Reddit posts and store them in MongoDB."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip posts that already exist in MongoDB.",
    )
    parser.add_argument(
        "--refresh-days",
        type=int,
        default=None,
        help="Allow re-fetch for posts newer than N days even if they exist in MongoDB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    posts, _ = run_pipeline(
        skip_existing=args.skip_existing,
        refresh_days=args.refresh_days,
    )
    if not posts:
        logger.info("No posts fetched.")
        return

    store = connect_from_config()
    try:
        post_count, comment_count = store.upsert_posts_and_comments(posts)
        logger.info(
            "Stored %s posts and %s comments in MongoDB.", post_count, comment_count
        )
    finally:
        store.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
