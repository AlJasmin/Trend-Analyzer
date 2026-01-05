from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple

from pymongo import MongoClient, ASCENDING, DESCENDING

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"

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

        def set_postfix(self, **kwargs):
            return None

    def tqdm(iterable=None, **kwargs):
        return _NullTqdm(iterable, **kwargs)


def load_settings(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import ruamel.yaml as ry

        with path.open("r", encoding="utf-8") as fp:
            return ry.YAML().load(fp) or {}
    except Exception:
        import yaml

        with path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}


def get_mongo_settings(settings: Mapping[str, Any]) -> Tuple[str, str]:
    cfg = settings.get("mongodb") or {}
    uri = cfg.get("uri")
    database = cfg.get("database")
    if not uri or not database:
        raise ValueError("MongoDB settings missing (mongodb.uri / mongodb.database).")
    return uri, database


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_epoch_seconds(value: Any) -> int:
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def snapshot_week_from_created_utc(created_utc: int) -> str:
    dt = datetime.fromtimestamp(int(created_utc), tz=timezone.utc)
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


class MongoStore:
    def __init__(self, uri: str, db_name: str) -> None:
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.posts = self.db["posts"]
        self.comments = self.db["comments"]

    def ensure_indexes(self) -> None:
        self.posts.create_index([("post_id", ASCENDING)], unique=True)
        self.posts.create_index([("subreddit", ASCENDING), ("created_utc", DESCENDING)])
        self.posts.create_index([("topic_id", ASCENDING), ("snapshot_week", ASCENDING)])
        self.posts.create_index([("snapshot_week", ASCENDING)])

        self.comments.create_index([("comment_id", ASCENDING)], unique=True)
        self.comments.create_index([("post_id", ASCENDING), ("created_utc", DESCENDING)])
        self.comments.create_index([("snapshot_week", ASCENDING)])

    def close(self) -> None:
        self.client.close()

    def upsert_post(self, post: Any) -> bool:
        doc = post.to_dict() if hasattr(post, "to_dict") else dict(post)
        post_id = doc.get("post_id") or doc.get("id")
        if not post_id:
            logger.warning("Skipping post without post_id.")
            return False

        created_utc = to_epoch_seconds(doc.get("created_utc"))
        snapshot_week = doc.get("snapshot_week") or snapshot_week_from_created_utc(created_utc)

        data = {
            "post_id": str(post_id),
            "subreddit": doc.get("subreddit", ""),
            "title": doc.get("title", ""),
            "selftext": doc.get("selftext", ""),
            "cleaned_selftext": doc.get("cleaned_selftext", ""),
            "topic_text": doc.get("topic_text", ""),
            "created_utc": created_utc,
            "score": int(doc.get("score") or 0),
            "num_comments": int(doc.get("num_comments") or 0),
            "embedding": doc.get("embedding"),
            "topic_id": doc.get("topic_id"),
            "topic_name": doc.get("topic_name"),
            "topic_description": doc.get("topic_description"),
            "stance_dist_weighted": doc.get("stance_dist_weighted"),
            "sentiment_dist_weighted": doc.get("sentiment_dist_weighted"),
            "polarization_score": doc.get("polarization_score"),
            "snapshot_week": snapshot_week,
            "updated_at": utc_now(),
        }

        update = {"$set": data, "$setOnInsert": {"created_at": utc_now()}}
        self.posts.update_one({"post_id": str(post_id)}, update, upsert=True)
        return True

    def upsert_comment(self, comment: Any) -> bool:
        doc = comment if isinstance(comment, dict) else dict(comment)
        comment_id = doc.get("comment_id") or doc.get("id")
        post_id = doc.get("post_id")
        if not comment_id or not post_id:
            logger.warning("Skipping comment without comment_id/post_id.")
            return False

        created_utc = to_epoch_seconds(doc.get("created_utc"))
        snapshot_week = doc.get("snapshot_week") or snapshot_week_from_created_utc(created_utc)

        comment_text = doc.get("comment_text") or doc.get("body", "")
        comment_text_clean = doc.get("comment_text_clean") or doc.get("body_clean", "")
        upvote_score = doc.get("upvote_score")
        if upvote_score is None:
            upvote_score = doc.get("score", 0)

        data = {
            "comment_id": str(comment_id),
            "post_id": str(post_id),
            "comment_text": comment_text,
            "comment_text_clean": comment_text_clean,
            "upvote_score": int(upvote_score or 0),
            "created_utc": created_utc,
            "sentiment_label": doc.get("sentiment_label"),
            "stance_label": doc.get("stance_label"),
            "weight": doc.get("weight"),
            "snapshot_week": snapshot_week,
            "updated_at": utc_now(),
        }

        update = {"$set": data, "$setOnInsert": {"created_at": utc_now()}}
        self.comments.update_one({"comment_id": str(comment_id)}, update, upsert=True)
        return True

    def upsert_posts_and_comments(self, posts: Iterable[Any], show_progress: bool = True) -> tuple[int, int]:
        post_count = 0
        comment_count = 0

        post_iter = posts
        post_bar = None
        comment_bar = None
        if show_progress:
            post_bar = tqdm(posts, desc="Save posts", unit="post")
            post_iter = post_bar
            comment_bar = tqdm(desc="Save comments", unit="comment", leave=False)

        for post in post_iter:
            if self.upsert_post(post):
                post_count += 1

            post_id = getattr(post, "post_id", None)
            comments = getattr(post, "comments", None)
            if comments is None:
                doc = post.to_dict() if hasattr(post, "to_dict") else dict(post)
                comments = doc.get("comments", [])
                if post_id is None:
                    post_id = doc.get("post_id") or doc.get("id")

            for comment in comments or []:
                if isinstance(comment, dict) and not comment.get("post_id"):
                    comment["post_id"] = post_id
                if self.upsert_comment(comment):
                    comment_count += 1
                    if comment_bar is not None:
                        comment_bar.update(1)

        if comment_bar is not None:
            comment_bar.close()
        if post_bar is not None:
            post_bar.close()

        return post_count, comment_count


def connect_from_config(config_path: Optional[Path] = None) -> MongoStore:
    settings = load_settings(config_path or CONFIG_PATH)
    uri, db_name = get_mongo_settings(settings)
    store = MongoStore(uri=uri, db_name=db_name)
    store.ensure_indexes()
    return store
