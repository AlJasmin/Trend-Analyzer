"""Basic cleaning utilities for Reddit items (normalization, remove duplicates, minimal text cleaning)."""

import re
from typing import Iterable, Union

try:
    from reddit.models import RedditPost
except ImportError:  # pragma: no cover - avoids circular issues when module path changes
    RedditPost = None

PostLike = Union[dict, "RedditPost"]


def _get_attr(post: PostLike, attr: str, default=""):
    if hasattr(post, attr):
        return getattr(post, attr, default) or default
    return post.get(attr, default) if isinstance(post, dict) else default


def _set_attr(post: PostLike, attr: str, value):
    if hasattr(post, attr):
        setattr(post, attr, value)
    elif isinstance(post, dict):
        post[attr] = value


def clean_post(post: PostLike) -> PostLike:
    """Normalize whitespace and attach a clean_text field for either dict or RedditPost inputs."""
    title = _get_attr(post, 'title', '')
    body = _get_attr(post, 'selftext', '')
    text = (title + '\n' + body).strip()
    text = re.sub(r'\s+', ' ', text)
    _set_attr(post, 'clean_text', text)
    return post


def clean_posts(posts: Iterable[PostLike]) -> list:
    """Deduplicate posts and normalize text."""
    seen = set()
    out = []
    for p in posts:
        post_id = _get_attr(p, 'post_id') or _get_attr(p, 'id')
        if not post_id:
            # Fall back to object identity to avoid dropping legitimate entries
            post_id = id(p)
        if post_id in seen:
            continue
        seen.add(post_id)
        out.append(clean_post(p))
    return out

if __name__ == '__main__':
    print('reddit_cleaner stub')
