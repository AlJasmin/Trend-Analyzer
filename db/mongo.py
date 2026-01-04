from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def snapshot_week_from_created_utc(created_utc: int) -> str:
    dt = datetime.fromtimestamp(int(created_utc), tz=timezone.utc)
    year, week, _ = dt.isocalendar()  # ISO week
    return f"{year}-W{week:02d}"


class MongoStore:
    """
    Minimal DB:
    - posts
    - comments
    """

    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "trend_analyzer"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

        self.posts: Collection = self.db["posts"]
        self.comments: Collection = self.db["comments"]

    def ensure_indexes(self) -> None:
        # posts
        self.posts.create_index([("post_id", ASCENDING)], unique=True)
        self.posts.create_index([("subreddit", ASCENDING), ("created_utc", DESCENDING)])
        self.posts.create_index([("snapshot_week", ASCENDING)])
        self.posts.create_index([("topic_id", ASCENDING), ("snapshot_week", ASCENDING)])

        # comments
        self.comments.create_index([("comment_id", ASCENDING)], unique=True)
        self.comments.create_index([("post_id", ASCENDING), ("created_utc", DESCENDING)])
        self.comments.create_index([("snapshot_week", ASCENDING)])
        self.comments.create_index([("sentiment_label", ASCENDING), ("snapshot_week", ASCENDING)])
        self.comments.create_index([("stance_label", ASCENDING), ("snapshot_week", ASCENDING)])

    # -------------------------
    # POSTS
    # -------------------------

    def upsert_post_base(self, post: Dict[str, Any]) -> None:
        """
        Insert/Update basic post fields.
        Required: post_id, subreddit, created_utc
        """
        post_id = str(post["post_id"])
        created_utc = int(post["created_utc"])

        doc = {
            "post_id": post_id,
            "subreddit": post.get("subreddit"),
            "title": post.get("title", ""),
            "selftext": post.get("selftext", ""),
            "created_utc": created_utc,
            "score": int(post.get("score", 0)),
            "num_comments": int(post.get("num_comments", 0)),
            "snapshot_week": post.get("snapshot_week") or snapshot_week_from_created_utc(created_utc),
            "updated_at": utc_now(),
        }

        # keep created_at stable
        self.posts.update_one(
            {"post_id": post_id},
            {
                "$set": doc,
                "$setOnInsert": {"created_at": utc_now()},
            },
            upsert=True,
        )

    def set_post_topic_and_embedding(
        self,
        post_id: str,
        topic_text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        topic_id: Optional[str] = None,
        topic_name: Optional[str] = None,
        topic_description: Optional[str] = None,
    ) -> None:
        """
        For topic modeling / embedding script.
        """
        update: Dict[str, Any] = {"updated_at": utc_now()}
        if topic_text is not None:
            update["topic_text"] = topic_text
        if embedding is not None:
            update["embedding"] = embedding
        if topic_id is not None:
            update["topic_id"] = topic_id
        if topic_name is not None:
            update["topic_name"] = topic_name
        if topic_description is not None:
            update["topic_description"] = topic_description

        self.posts.update_one({"post_id": str(post_id)}, {"$set": update})

    def set_post_aggregates(
        self,
        post_id: str,
        stance_dist_weighted: Optional[Dict[str, float]] = None,
        sentiment_dist_weighted: Optional[Dict[str, float]] = None,
        polarization_score: Optional[float] = None,
        snapshot_week: Optional[str] = None,
    ) -> None:
        """
        For aggregation script (after comments are processed).
        """
        update: Dict[str, Any] = {"updated_at": utc_now()}
        if stance_dist_weighted is not None:
            update["stance_dist_weighted"] = stance_dist_weighted
        if sentiment_dist_weighted is not None:
            update["sentiment_dist_weighted"] = sentiment_dist_weighted
        if polarization_score is not None:
            update["polarization_score"] = float(polarization_score)
        if snapshot_week is not None:
            update["snapshot_week"] = snapshot_week

        self.posts.update_one({"post_id": str(post_id)}, {"$set": update})

    # -------------------------
    # COMMENTS
    # -------------------------

    def upsert_comment_base(self, comment: Dict[str, Any]) -> None:
        """
        Insert/Update base comment fields.
        Required: comment_id, post_id, created_utc
        """
        comment_id = str(comment["comment_id"])
        post_id = str(comment["post_id"])
        created_utc = int(comment["created_utc"])

        doc = {
            "comment_id": comment_id,
            "post_id": post_id,
            "body_clean": comment.get("body_clean", ""),
            "score": int(comment.get("score", 0)),
            "created_utc": created_utc,
            "snapshot_week": comment.get("snapshot_week") or snapshot_week_from_created_utc(created_utc),
            "updated_at": utc_now(),
        }

        self.comments.update_one(
            {"comment_id": comment_id},
            {"$set": doc, "$setOnInsert": {"created_at": utc_now()}},
            upsert=True,
        )

    def set_comment_labels(
        self,
        comment_id: str,
        sentiment_label: Optional[str] = None,
        stance_label: Optional[str] = None,
        weight: Optional[float] = None,
        snapshot_week: Optional[str] = None,
    ) -> None:
        """
        For sentiment/stance script.
        """
        update: Dict[str, Any] = {"updated_at": utc_now()}
        if sentiment_label is not None:
            update["sentiment_label"] = sentiment_label
        if stance_label is not None:
            update["stance_label"] = stance_label
        if weight is not None:
            update["weight"] = float(weight)
        if snapshot_week is not None:
            update["snapshot_week"] = snapshot_week

        self.comments.update_one({"comment_id": str(comment_id)}, {"$set": update})


def connect_from_env() -> MongoStore:
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "trend_analyzer")
    store = MongoStore(uri=uri, db_name=db_name)
    store.ensure_indexes()
    return store
