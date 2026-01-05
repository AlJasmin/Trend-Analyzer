from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple

from pymongo import MongoClient, ASCENDING, DESCENDING

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
NUMERIC_TYPES = ["int", "long", "double", "decimal"]
STRING_OR_NULL = ["string", "null"]
ARRAY_OR_NULL = ["array", "null"]
OBJECT_OR_NULL = ["object", "null"]
NUMERIC_OR_NULL = NUMERIC_TYPES + ["null"]


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


def ensure_collection(db, name: str, validator: dict):
    if name in db.list_collection_names():
        db.command(
            {
                "collMod": name,
                "validator": validator,
                "validationLevel": "moderate",
                "validationAction": "warn",
            }
        )
        return db[name]
    return db.create_collection(
        name,
        validator=validator,
        validationLevel="moderate",
        validationAction="warn",
    )


def build_validator(required: list, properties: dict) -> dict:
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": required,
            "properties": properties,
            "additionalProperties": True,
        }
    }


def main() -> None:
    settings = load_settings(CONFIG_PATH)
    uri, db_name = get_mongo_settings(settings)

    client = MongoClient(uri)
    db = client[db_name]

    posts_validator = build_validator(
        ["post_id", "subreddit", "created_utc", "snapshot_week"],
        {
            "post_id": {"bsonType": "string"},
            "subreddit": {"bsonType": "string"},
            "title": {"bsonType": STRING_OR_NULL},
            "selftext": {"bsonType": STRING_OR_NULL},
            "cleaned_selftext": {"bsonType": STRING_OR_NULL},
            "topic_text": {"bsonType": STRING_OR_NULL},
            "link_flair_text": {"bsonType": STRING_OR_NULL},
            "created_utc": {"bsonType": ["int", "long"]},
            "score": {"bsonType": NUMERIC_OR_NULL},
            "upvote_count": {"bsonType": NUMERIC_OR_NULL},
            "upvote_ratio": {"bsonType": NUMERIC_OR_NULL},
            "num_comments": {"bsonType": NUMERIC_OR_NULL},
            "embedding": {"bsonType": ARRAY_OR_NULL},
            "topic_id": {"bsonType": STRING_OR_NULL},
            "topic_name": {"bsonType": STRING_OR_NULL},
            "topic_description": {"bsonType": STRING_OR_NULL},
            "stance_dist_weighted": {"bsonType": OBJECT_OR_NULL},
            "sentiment_dist_weighted": {"bsonType": OBJECT_OR_NULL},
            "polarization_score": {"bsonType": NUMERIC_OR_NULL},
            "snapshot_week": {"bsonType": "string"},
        },
    )
    
    comments_validator = build_validator(
        ["comment_id", "post_id", "created_utc", "snapshot_week"],
        {
            "comment_id": {"bsonType": "string"},
            "post_id": {"bsonType": "string"},
            "comment_text": {"bsonType": STRING_OR_NULL},
            "comment_text_clean": {"bsonType": STRING_OR_NULL},
            "upvote_score": {"bsonType": NUMERIC_OR_NULL},
            "created_utc": {"bsonType": ["int", "long"]},
            "sentiment_label": {"bsonType": STRING_OR_NULL},
            "stance_label": {"bsonType": STRING_OR_NULL},
            "weight": {"bsonType": NUMERIC_OR_NULL},
            "snapshot_week": {"bsonType": "string"},
        },
    )

    posts = ensure_collection(db, "posts", posts_validator)
    comments = ensure_collection(db, "comments", comments_validator)

    posts.create_index([("post_id", ASCENDING)], unique=True)
    posts.create_index([("subreddit", ASCENDING), ("created_utc", DESCENDING)])
    posts.create_index([("topic_id", ASCENDING), ("snapshot_week", ASCENDING)])
    posts.create_index([("snapshot_week", ASCENDING)])

    comments.create_index([("comment_id", ASCENDING)], unique=True)
    comments.create_index([("post_id", ASCENDING), ("created_utc", DESCENDING)])
    comments.create_index([("snapshot_week", ASCENDING)])

    client.close()


if __name__ == "__main__":
    main()


