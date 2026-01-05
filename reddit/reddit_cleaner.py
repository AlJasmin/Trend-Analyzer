"""Basic cleaning utilities for Reddit items (normalization, remove duplicates, minimal text cleaning)."""

from typing import Iterable, Union

from processing.text_cleaning import clean_text

try:
    from reddit.models import RedditPost, RedditComment
except ImportError:  # pragma: no cover - avoids circular issues when module path changes
    RedditPost = None
    RedditComment = None

PostLike = Union[dict, "RedditPost"]
CommentLike = Union[dict, "RedditComment"]


def _get_attr(post: PostLike, attr: str, default=""):
    if hasattr(post, attr):
        return getattr(post, attr, default) or default
    return post.get(attr, default) if isinstance(post, dict) else default


def _set_attr(post: PostLike, attr: str, value):
    if hasattr(post, attr):
        setattr(post, attr, value)
    elif isinstance(post, dict):
        post[attr] = value


def build_topic_text(title: str, selftext: str, is_self: bool) -> str:
    cleaned_title = clean_text(title)
    if is_self:
        cleaned_selftext = clean_text(selftext)
        if cleaned_selftext:
            return (cleaned_title + "\n" + cleaned_selftext).strip()
    return cleaned_title


def clean_post(post: PostLike) -> PostLike:
    """Attach cleaned fields and topic_text for dict or RedditPost inputs."""
    title = _get_attr(post, "title", "")
    selftext = _get_attr(post, "selftext", "")
    is_self = bool(_get_attr(post, "is_self", False))

    cleaned_selftext = clean_text(selftext) if is_self else ""
    topic_text = build_topic_text(title, selftext, is_self)

    _set_attr(post, "cleaned_selftext", cleaned_selftext)
    _set_attr(post, "topic_text", topic_text)
    _set_attr(post, "clean_text", topic_text)
    return post


def clean_comment(comment: CommentLike) -> CommentLike:
    body = _get_attr(comment, "body", "") or _get_attr(comment, "comment_text", "")
    cleaned = clean_text(body)
    _set_attr(comment, "body_clean", cleaned)
    _set_attr(comment, "comment_text_clean", cleaned)
    return comment


def clean_posts(posts: Iterable[PostLike]) -> list:
    """Deduplicate posts and attach cleaned fields."""
    seen = set()
    out = []
    for p in posts:
        post_id = _get_attr(p, "post_id") or _get_attr(p, "id")
        if not post_id:
            post_id = id(p)
        if post_id in seen:
            continue
        seen.add(post_id)
        out.append(clean_post(p))
    return out


def clean_comments(comments: Iterable[CommentLike]) -> list:
    """Attach cleaned fields for comments."""
    return [clean_comment(c) for c in comments]

if __name__ == '__main__':
    print('reddit_cleaner stub')
