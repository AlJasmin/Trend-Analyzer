"""Config-driven Reddit scraping pipeline."""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from reddit.reddit_client import RedditClient  # type: ignore
    from reddit.fetchers import PostFetcher, CommentFetcher  # type: ignore
    from reddit.post_filter import PostFilter  # type: ignore
    from reddit.enrichers import (
        ImageEnricher,
        CommentEnricher,
    )  # type: ignore
    from reddit.reddit_cleaner import clean_posts  # type: ignore
    from utils.json_utils import write_json  # type: ignore
except ImportError:
    # Allow running via `python reddit/reddit_scraper.py`
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from reddit.reddit_client import RedditClient  # type: ignore
    from reddit.fetchers import PostFetcher, CommentFetcher  # type: ignore
    from reddit.post_filter import PostFilter  # type: ignore
    from reddit.enrichers import (
        ImageEnricher,
        CommentEnricher,
    )  # type: ignore
    from reddit.reddit_cleaner import clean_posts  # type: ignore
    from utils.json_utils import write_json  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_FETCH = {"feed": "top", "time_filter": "day", "limit": 25}

DEFAULT_PIPELINE = {
    "subreddits": [
        {
            "name": "memes",
            "category": "memes",
            "fetch": {"feed": "top", "time_filter": "day", "limit": 25},
        }
    ],
    "filters": {
        "min_score": 0,
        "recent_days": 7,
        "allowed_categories": [],
        "excluded_categories": [],
        "sort_by": "score",
        "descending": True,
        "top_n": 50,
    },
    "subreddit_list_file": None,
    "fetch_defaults": DEFAULT_FETCH.copy(),
    "comments": {
        "enabled": True,
        "fetch_mode": "smart",
        "limit": 50,
        "log_count": False,
    },
    "logging": {
        "show_post_timestamp": False,
        "show_comment_timestamp": False,
        "comment_preview_count": 0,
    },
    "enrichers": {
        "images": {"enabled": False, "embedder_model": "clip"},
    },
}


def load_config() -> Dict[str, Any]:
    """Load the YAML settings file."""
    try:
        import ruamel.yaml as ry

        with open(CONFIG, "r", encoding="utf-8") as f:
            return ry.YAML().load(f) or {}
    except FileNotFoundError:
        logger.warning("Config file %s not found. Using defaults.", CONFIG)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to load config: %s", exc)
    return {}


def _load_subreddits_from_file(file_path: Optional[str], default_category: str = "general") -> List[Dict[str, str]]:
    if not file_path:
        return []
    path = Path(file_path)
    if not path.is_absolute():
        path = (REPO_ROOT / file_path).resolve()
    if not path.exists():
        logger.warning("Subreddit list file %s not found.", path)
        return []

    pattern = re.compile(r"^(?P<name>[^()]+?)(?:\s*\((?P<category>[^)]+)\))?$")
    entries: List[Dict[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            logger.warning("Skipping malformed subreddit line: %s", raw_line.strip())
            continue
        name = match.group("name").strip()
        if not name:
            continue
        category = (match.group("category") or default_category).strip()
        entries.append({"name": name, "category": category})
    return entries


def _normalize_subreddits(
    raw_subreddits: Optional[List[Dict[str, Any]]],
    file_entries: List[Dict[str, Any]],
    fetch_defaults: Dict[str, Any],
    default_category: str = "general",
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen = set()

    def _add_entry(name: Optional[str], category: Optional[str], fetch_override: Optional[Dict[str, Any]] = None):
        if not name:
            return
        key = name.lower()
        if key in seen:
            return
        seen.add(key)
        fetch_cfg = fetch_defaults.copy()
        if fetch_override:
            fetch_cfg.update(fetch_override)
        normalized.append(
            {
                "name": name,
                "category": (category or default_category),
                "fetch": fetch_cfg,
            }
        )

    for entry in raw_subreddits or []:
        name = (entry.get("name") or "").strip()
        if not name:
            logger.warning("Skipping subreddit entry without a name: %s", entry)
            continue
        category = entry.get("category", default_category)
        fetch_override = entry.get("fetch") or {}
        _add_entry(name, category, fetch_override)

    for entry in file_entries:
        _add_entry(entry.get("name"), entry.get("category", default_category))

    if normalized:
        return normalized

    fallback: List[Dict[str, Any]] = []
    for entry in DEFAULT_PIPELINE["subreddits"]:
        fallback.append(
            {
                "name": entry.get("name"),
                "category": entry.get("category", default_category),
                "fetch": {**fetch_defaults, **(entry.get("fetch") or {})},
            }
        )
    return fallback


def _merge_section(user_cfg: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = defaults.copy()
    if not user_cfg:
        return merged
    for key, value in user_cfg.items():
        if isinstance(value, dict) and isinstance(defaults.get(key), dict):
            merged[key] = _merge_section(value, defaults[key])
        else:
            merged[key] = value
    return merged


def _format_timestamp(value) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (int, float)):
        return datetime.utcfromtimestamp(value).isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - fallback
            pass
    if value is None:
        return "unknown"
    return str(value)


def build_pipeline_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    pipeline_cfg = cfg.get("reddit_pipeline") or {}
    fetch_defaults = _merge_section(pipeline_cfg.get("fetch_defaults"), DEFAULT_PIPELINE["fetch_defaults"])
    subreddit_list_file = pipeline_cfg.get("subreddit_list_file") or DEFAULT_PIPELINE.get("subreddit_list_file")
    file_entries = _load_subreddits_from_file(subreddit_list_file)
    subreddits = _normalize_subreddits(pipeline_cfg.get("subreddits"), file_entries, fetch_defaults)
    return {
        "subreddits": subreddits,
        "filters": _merge_section(pipeline_cfg.get("filters"), DEFAULT_PIPELINE["filters"]),
        "comments": _merge_section(pipeline_cfg.get("comments"), DEFAULT_PIPELINE["comments"]),
        "logging": _merge_section(pipeline_cfg.get("logging"), DEFAULT_PIPELINE["logging"]),
        "enrichers": _merge_section(pipeline_cfg.get("enrichers"), DEFAULT_PIPELINE["enrichers"]),
        "fetch_defaults": fetch_defaults,
        "subreddit_list_file": subreddit_list_file,
    }


def apply_filters(posts, filters_cfg):
    post_filter = PostFilter()
    if filters_cfg.get("min_score") is not None:
        posts = post_filter.filter_by_score(posts, int(filters_cfg["min_score"]))
    if filters_cfg.get("recent_days"):
        posts = post_filter.filter_by_recency(posts, int(filters_cfg["recent_days"]))
    allowed = filters_cfg.get("allowed_categories") or []
    if allowed:
        posts = post_filter.filter_by_category(posts, allowed)
    excluded = filters_cfg.get("excluded_categories") or []
    if excluded:
        posts = post_filter.exclude_by_category(posts, excluded)
    posts = post_filter.deduplicate(posts)
    sort_by = (filters_cfg.get("sort_by") or "score").lower()
    descending = bool(filters_cfg.get("descending", True))
    if sort_by == "score":
        posts = post_filter.sort_by_score(posts, descending=descending)
    elif sort_by == "recency":
        posts = post_filter.sort_by_recency(posts, descending=descending)
    top_n = filters_cfg.get("top_n")
    if isinstance(top_n, int) and top_n > 0:
        posts = post_filter.get_top_n(posts, top_n)
    return posts


def _get_attr(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_comment_for_json(comment: Any) -> Dict[str, Any]:
    def fmt_ts(val):
        return _format_timestamp(val) if val else ""

    def val(key, default=""):
        return _get_attr(comment, key, default)

    return {
        "comment_id": val("comment_id") or val("id") or "",
        "author": val("author", ""),
        "body": val("body", ""),
        "score": val("score", 0) or 0,
        "created_utc": fmt_ts(val("created_utc")),
        "first_seen": fmt_ts(val("first_seen")),
        "last_updated": fmt_ts(val("last_updated")),
        "score_history": val("score_history", []) or [],
        "historical": bool(val("historical", False)),
        "dropped_from_top": fmt_ts(val("dropped_from_top")),
        "clean_text": val("clean_text", ""),
        "sentiment": val("sentiment", {}) or {},
        "embedding": val("embedding", []) or [],
    }


def _normalize_post_for_json(post: Any) -> Dict[str, Any]:
    def fmt_ts(val):
        return _format_timestamp(val) if val else ""

    comments_raw = _get_attr(post, "comments", []) or []
    comments = [_normalize_comment_for_json(c) for c in comments_raw]

    return {
        "post_id": _get_attr(post, "post_id", "") or _get_attr(post, "id", ""),
        "title": _get_attr(post, "title", ""),
        "author": _get_attr(post, "author", ""),
        "subreddit": _get_attr(post, "subreddit", ""),
        "category": _get_attr(post, "category", ""),
        "created_utc": fmt_ts(_get_attr(post, "created_utc", "")),
        "score": _get_attr(post, "score", 0) or 0,
        "num_comments": _get_attr(post, "num_comments", 0) or 0,
        "upvote_ratio": _get_attr(post, "upvote_ratio", 0.0) or 0.0,
        "permalink": _get_attr(post, "permalink", ""),
        "url": _get_attr(post, "url", ""),
        "is_self": bool(_get_attr(post, "is_self", False)),
        "selftext": _get_attr(post, "selftext", ""),
        "link_flair_text": _get_attr(post, "link_flair_text", "") or "",
        "comments": comments,
        "photo_parse": _get_attr(post, "photo_parse", ""),
        "historical_metrics": _get_attr(post, "historical_metrics", []) or [],
        "last_updated": fmt_ts(_get_attr(post, "last_updated", "")),
        "clean_text": _get_attr(post, "clean_text", ""),
        "sentiment": _get_attr(post, "sentiment", {}) or {},
        "embedding": _get_attr(post, "embedding", []) or [],
    }


def export_posts_to_json(posts: List[Any]) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "data" / "raw"
    output_file = output_dir / f"reddit_posts_{timestamp}.json"
    payload = [_normalize_post_for_json(p) for p in posts]
    write_json(output_file, payload)
    logger.info("Saved %s posts to %s", len(payload), output_file)
    return output_file


def build_enrichers(pipeline_cfg, comment_fetcher):
    enrich_cfg = pipeline_cfg.get("enrichers", {})
    comments_cfg = pipeline_cfg.get("comments", {})

    image_enricher = None
    image_cfg = enrich_cfg.get("images", {})
    if image_cfg.get("enabled"):
        image_enricher = ImageEnricher(embedder_model=image_cfg.get("embedder_model", "clip"))
        logger.info("Image enrichment enabled via YAML config.")

    comment_enricher = None
    if comments_cfg.get("enabled", True):
        comment_enricher = CommentEnricher(comment_fetcher)

    return image_enricher, comment_enricher, comments_cfg


def enrich_posts(posts, pipeline_cfg, comment_fetcher):
    image_enricher, comment_enricher, comments_cfg = build_enrichers(pipeline_cfg, comment_fetcher)
    for post in posts:
        if image_enricher:
            image_enricher.enrich_post(post)
        if comment_enricher:
            comment_enricher.enrich_post(
                post,
                fetch_mode=comments_cfg.get("fetch_mode", "smart"),
                limit=comments_cfg.get("limit"),
            )
    return posts


def run_pipeline() -> tuple[List[Any], Dict[str, Any]]:
    cfg = load_config()
    pipeline_cfg = build_pipeline_config(cfg)
    client = RedditClient(config=cfg)
    post_fetcher = PostFetcher(client)
    comment_fetcher = CommentFetcher(client)

    posts = []
    for subreddit_cfg in pipeline_cfg["subreddits"]:
        fetch_cfg = subreddit_cfg["fetch"]
        fetched = post_fetcher.fetch_posts(
            subreddit=subreddit_cfg["name"],
            feed=fetch_cfg.get("feed", "top"),
            time_filter=fetch_cfg.get("time_filter", "day"),
            limit=int(fetch_cfg.get("limit", 25)),
        )
        category = subreddit_cfg.get("category")
        if category:
            for post in fetched:
                post.category = category
        posts.extend(fetched)

    if not posts:
        logger.info("No posts fetched for configured subreddits.")
        return [], pipeline_cfg

    posts = apply_filters(posts, pipeline_cfg["filters"])
    posts = enrich_posts(posts, pipeline_cfg, comment_fetcher)
    posts = clean_posts(posts)
    return posts, pipeline_cfg


def main():
    posts, pipeline_cfg = run_pipeline()
    logger.info("Pipeline completed with %s posts.", len(posts))
    comments_cfg = pipeline_cfg.get("comments", {})
    log_comment_count = bool(comments_cfg.get("log_count"))
    logging_cfg = pipeline_cfg.get("logging", {})
    show_post_ts = bool(logging_cfg.get("show_post_timestamp"))
    show_comment_ts = bool(logging_cfg.get("show_comment_timestamp"))
    comment_preview_count = int(logging_cfg.get("comment_preview_count") or 0)

    for post in posts:
        logger.info("r/%s [%s]: %s (score=%s)", post.subreddit, post.category, post.title, post.score)
        if show_post_ts:
            logger.info("  created_at=%s", _format_timestamp(getattr(post, "created_utc", None)))

        comments = getattr(post, "comments", []) or []
        if log_comment_count:
            logger.info("  comment_count=%s", len(comments))

        if show_comment_ts and comment_preview_count > 0:
            for idx, comment in enumerate(comments[:comment_preview_count], start=1):
                logger.info(
                    "    ↳ #%s [%s] %s: %s",
                    idx,
                    _format_timestamp(comment.get("created_utc")),
                    comment.get("author", "[unknown]"),
                    (comment.get("body") or "").strip(),
                )

    if posts:
        export_posts_to_json(posts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
