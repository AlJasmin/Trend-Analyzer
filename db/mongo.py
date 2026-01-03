from __future__ import annotations

import os
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database


# -------------------------
# Utils
# -------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def make_source_ref(type_: str, id_: str, field: Optional[str] = None) -> Dict[str, Any]:
    ref = {"type": type_, "id": id_}
    if field:
        ref["field"] = field
    return ref


# -------------------------
# Mongo wrapper
# -------------------------

class MongoStore:
    """
    Single entry-point for all scripts:
    - Reddit ingester writes to reddit_posts (+ images links)
    - Web scraper writes to web_articles (+ images links)
    - Sentiment script writes to sentiments
    - Topic modeling writes to topics + trend_daily
    etc.
    """

    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db: Database = self.client[db_name]

        # Collections
        self.reddit_posts: Collection = self.db["reddit_posts"]
        self.web_articles: Collection = self.db["web_articles"]
        self.images: Collection = self.db["images"]
        self.text_embeddings: Collection = self.db["text_embeddings"]
        self.image_embeddings: Collection = self.db["image_embeddings"]
        self.ocr_texts: Collection = self.db["ocr_texts"]
        self.topics: Collection = self.db["topics"]
        self.sentiments: Collection = self.db["sentiments"]
        self.llm_outputs: Collection = self.db["llm_outputs"]
        self.ml_predictions: Collection = self.db["ml_predictions"]
        self.trend_daily: Collection = self.db["trend_daily"]

    # -------------------------
    # Indexes (call once at start)
    # -------------------------

    def ensure_indexes(self) -> None:
        # reddit_posts
        self.reddit_posts.create_index([("subreddit", ASCENDING), ("created_utc", DESCENDING)])
        self.reddit_posts.create_index([("created_utc", DESCENDING)])
        self.reddit_posts.create_index([("permalink", ASCENDING)], unique=False)

        # web_articles
        self.web_articles.create_index([("url", ASCENDING)], unique=True)
        self.web_articles.create_index([("domain", ASCENDING), ("published_at", DESCENDING)])
        self.web_articles.create_index([("published_at", DESCENDING)])

        # images
        self.images.create_index([("image_url", ASCENDING)], unique=True)
        self.images.create_index([("source_ref.type", ASCENDING), ("source_ref.id", ASCENDING)])
        self.images.create_index([("kind", ASCENDING)])

        # embeddings
        self.text_embeddings.create_index([("source_ref.type", ASCENDING), ("source_ref.id", ASCENDING)])
        self.image_embeddings.create_index([("image_id", ASCENDING)])

        # ocr
        self.ocr_texts.create_index([("image_id", ASCENDING)])

        # topics, sentiments, llm, ml
        self.topics.create_index([("source_ref.type", ASCENDING), ("source_ref.id", ASCENDING)])
        self.topics.create_index([("topic_id", ASCENDING), ("created_at", DESCENDING)])

        self.sentiments.create_index([("source_ref.type", ASCENDING), ("source_ref.id", ASCENDING)])
        self.sentiments.create_index([("created_at", DESCENDING)])

        self.llm_outputs.create_index([("source_ref.type", ASCENDING), ("source_ref.id", ASCENDING)])
        self.llm_outputs.create_index([("output_type", ASCENDING), ("created_at", DESCENDING)])

        self.ml_predictions.create_index([("source_ref.type", ASCENDING), ("source_ref.id", ASCENDING)])
        self.ml_predictions.create_index([("model", ASCENDING), ("created_at", DESCENDING)])

        # trend_daily
        self.trend_daily.create_index([("topic_id", ASCENDING), ("date", ASCENDING)], unique=True)
        self.trend_daily.create_index([("date", ASCENDING)])

    # -------------------------
    # Insert / Upsert: Reddit + Web
    # -------------------------

    def upsert_reddit_post(self, post: Dict[str, Any]) -> str:
        """
        Expects post dict with at least:
          id, subreddit, created_utc, title, selftext, url/permalink
        Uses _id = post["id"] (string).
        """
        _id = str(post["id"])
        doc = {**post}
        doc["_id"] = _id
        doc.setdefault("source", "reddit")
        doc.setdefault("ingested_at", utc_now())

        self.reddit_posts.update_one({"_id": _id}, {"$set": doc}, upsert=True)
        return _id

    def upsert_web_article(self, article: Dict[str, Any]) -> str:
        """
        Expects article with at least url.
        Uses _id = sha256(url) for stable ID.
        """
        url = article["url"]
        _id = sha256_str(url)
        doc = {**article}
        doc["_id"] = _id
        doc.setdefault("source", "web")
        doc.setdefault("ingested_at", utc_now())
        doc.setdefault("domain", self._extract_domain(url))

        self.web_articles.update_one({"_id": _id}, {"$set": doc}, upsert=True)
        return _id

    def _extract_domain(self, url: str) -> str:
        # simple domain extraction (no external deps)
        url2 = url.replace("https://", "").replace("http://", "")
        return url2.split("/")[0].lower()

    # -------------------------
    # Images
    # -------------------------

    def upsert_image_link(
        self,
        source_ref: Dict[str, Any],
        image_url: str,
        kind: str = "unknown",
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Stores image metadata + link. Keeps images separate.
        _id = sha256(image_url)
        """
        _id = sha256_str(image_url)
        doc = {
            "_id": _id,
            "source_ref": source_ref,
            "image_url": image_url,
            "kind": kind,
            "download_status": "pending",
            "created_at": utc_now(),
        }
        if meta:
            doc.update(meta)

        self.images.update_one({"_id": _id}, {"$set": doc}, upsert=True)
        return _id

    # -------------------------
    # OCR
    # -------------------------

    def upsert_ocr_text(
        self,
        image_id: str,
        text: str,
        engine: str,
        language: str = "en",
        confidence_avg: Optional[float] = None,
        version: str = "1",
    ) -> str:
        _id = sha256_str(f"{image_id}|{engine}|{version}")
        doc = {
            "_id": _id,
            "image_id": image_id,
            "engine": engine,
            "version": version,
            "text": text,
            "language": language,
            "confidence_avg": confidence_avg,
            "created_at": utc_now(),
        }
        self.ocr_texts.update_one({"_id": _id}, {"$set": doc}, upsert=True)
        return _id

    # -------------------------
    # Embeddings
    # -------------------------

    def upsert_text_embedding(
        self,
        source_ref: Dict[str, Any],
        model: str,
        vector: List[float],
    ) -> str:
        dim = len(vector)
        key = f"{source_ref['type']}|{source_ref['id']}|{source_ref.get('field','')}|{model}"
        _id = sha256_str(key)
        doc = {
            "_id": _id,
            "source_ref": source_ref,
            "model": model,
            "dim": dim,
            "vector": vector,
            "created_at": utc_now(),
        }
        self.text_embeddings.update_one({"_id": _id}, {"$set": doc}, upsert=True)
        return _id

    def upsert_image_embedding(
        self,
        image_id: str,
        model: str,
        vector: List[float],
    ) -> str:
        dim = len(vector)
        _id = sha256_str(f"{image_id}|{model}")
        doc = {
            "_id": _id,
            "image_id": image_id,
            "model": model,
            "dim": dim,
            "vector": vector,
            "created_at": utc_now(),
        }
        self.image_embeddings.update_one({"_id": _id}, {"$set": doc}, upsert=True)
        return _id

    # -------------------------
    # Topics / Sentiment / LLM / ML
    # -------------------------

    def upsert_topic_result(
        self,
        source_ref: Dict[str, Any],
        method: str,
        version: str,
        topic_id: str,
        topic_label: Optional[str] = None,
        topic_prob: Optional[float] = None,
        keywords: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        _id = sha256_str(f"{source_ref['type']}|{source_ref['id']}|{source_ref.get('field','')}|{method}|{version}")
        doc = {
            "_id": _id,
            "source_ref": source_ref,
            "method": method,
            "version": version,
            "topic_id": topic_id,
            "topic_label": topic_label,
            "topic_prob": topic_prob,
            "keywords": keywords or [],
            "created_at": utc_now(),
        }
        if extra:
            doc.update(extra)

        self.topics.update_one({"_id": _id}, {"$set": doc}, upsert=True)
        return _id

    def upsert_sentiment(
        self,
        source_ref: Dict[str, Any],
        model: str,
        version: str,
        label: str,
        score: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        _id = sha256_str(f"{source_ref['type']}|{source_ref['id']}|{source_ref.get('field','')}|{model}|{version}")
        doc = {
            "_id": _id,
            "source_ref": source_ref,
            "model": model,
            "version": version,
            "label": label,
            "score": float(score),
            "created_at": utc_now(),
        }
        if extra:
            doc.update(extra)

        self.sentiments.update_one({"_id": _id}, {"$set": doc}, upsert=True)
        return _id

    def upsert_llm_output(
        self,
        source_ref: Dict[str, Any],
        model: str,
        prompt_hash: str,
        output_type: str,
        text: str,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        _id = sha256_str(f"{source_ref['type']}|{source_ref['id']}|{source_ref.get('field','')}|{model}|{prompt_hash}|{output_type}")
        doc = {
            "_id": _id,
            "source_ref": source_ref,
            "model": model,
            "prompt_hash": prompt_hash,
            "output_type": output_type,
            "text": text,
            "json": json_data,
            "created_at": utc_now(),
        }
        self.llm_outputs.update_one({"_id": _id}, {"$set": doc}, upsert=True)
        return _id

    def upsert_ml_prediction(
        self,
        source_ref: Dict[str, Any],
        model: str,
        version: str,
        prediction: Dict[str, Any],
        features: Optional[Dict[str, Any]] = None,
    ) -> str:
        _id = sha256_str(f"{source_ref['type']}|{source_ref['id']}|{source_ref.get('field','')}|{model}|{version}")
        doc = {
            "_id": _id,
            "source_ref": source_ref,
            "model": model,
            "version": version,
            "prediction": prediction,
            "features": features or {},
            "created_at": utc_now(),
        }
        self.ml_predictions.update_one({"_id": _id}, {"$set": doc}, upsert=True)
        return _id

    # -------------------------
    # Trend time-series (daily)
    # -------------------------

    def inc_trend_daily(self, topic_id: str, date_str: str, inc: int = 1, sources: Optional[Dict[str, int]] = None) -> None:
        """
        date_str: "2026-01-03" etc.
        Upsert counter doc and increment count.
        """
        update = {
            "$inc": {"count": int(inc)},
            "$set": {"updated_at": utc_now()},
            "$setOnInsert": {"topic_id": topic_id, "date": date_str, "count": 0},
        }
        if sources:
            for k, v in sources.items():
                update["$inc"][f"sources.{k}"] = int(v)

        self.trend_daily.update_one({"topic_id": topic_id, "date": date_str}, update, upsert=True)


# -------------------------
# Factory (ENV-friendly)
# -------------------------

def connect_from_env() -> MongoStore:
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "trend_analyzer")
    store = MongoStore(uri=uri, db_name=db_name)
    store.ensure_indexes()
    return store
