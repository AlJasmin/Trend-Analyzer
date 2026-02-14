"""
Reddit Data Models

This module defines data models for Reddit posts and comments using dataclasses.
Provides a unified interface for converting between PRAW objects and dictionaries.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class RedditComment:
    """Data model for a Reddit comment."""

    comment_id: str
    author: str
    created_utc: datetime
    score: int
    body: str
    post_id: str = ""
    body_clean: str = ""
    sentiment_label: Optional[str] = None
    stance_label: Optional[str] = None
    weight: Optional[float] = None
    snapshot_week: Optional[str] = None

    # Tracking fields (added by database merge logic)
    first_seen: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    score_history: List[Dict[str, Any]] = field(default_factory=list)
    historical: bool = False
    dropped_from_top: Optional[datetime] = None

    @classmethod
    def from_praw(cls, comment, post_id: str = "") -> "RedditComment":
        """
        Create a RedditComment from a PRAW comment object.

        Args:
            comment: PRAW comment object

        Returns:
            RedditComment instance
        """
        return cls(
            comment_id=comment.id,
            author=str(comment.author) if comment.author else "[deleted]",
            created_utc=datetime.fromtimestamp(comment.created_utc),
            score=comment.score,
            body=comment.body,
            post_id=post_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for database storage.

        Returns:
            Dictionary representation
        """
        return {
            "comment_id": self.comment_id,
            "post_id": self.post_id,
            "author": self.author,
            "created_utc": self.created_utc,
            "score": self.score,
            "body": self.body,
            "body_clean": self.body_clean,
            "sentiment_label": self.sentiment_label,
            "stance_label": self.stance_label,
            "weight": self.weight,
            "snapshot_week": self.snapshot_week,
            "first_seen": self.first_seen,
            "last_updated": self.last_updated,
            "score_history": self.score_history,
            "historical": self.historical,
            "dropped_from_top": self.dropped_from_top,
        }


@dataclass
class RedditPost:
    """Data model for a Reddit post."""

    post_id: str
    title: str
    author: str
    created_utc: datetime
    score: int
    upvote_ratio: float
    num_comments: int
    permalink: str
    url: str
    is_self: bool
    selftext: str
    subreddit: str
    link_flair_text: Optional[str]
    category: str

    cleaned_selftext: str = ""
    topic_text: str = ""
    embedding: Optional[List[float]] = None
    topic_id: Optional[str] = None
    topic_name: Optional[str] = None
    topic_description: Optional[str] = None
    stance_dist_weighted: Optional[Dict[str, float]] = None
    sentiment_dist_weighted: Optional[Dict[str, float]] = None
    polarization_score: Optional[float] = None
    snapshot_week: Optional[str] = None

    # Optional enrichment fields
    comments: List[Dict[str, Any]] = field(default_factory=list)
    photo_parse: Optional[str] = None

    # Database tracking fields
    historical_metrics: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: Optional[datetime] = None

    @classmethod
    def from_praw(cls, post, category: str = "general") -> "RedditPost":
        """
        Create a RedditPost from a PRAW submission object.

        Args:
            post: PRAW submission object
            category: Post category (default: "general")

        Returns:
            RedditPost instance
        """
        return cls(
            post_id=post.id,
            title=post.title,
            author=str(post.author) if post.author else "[deleted]",
            created_utc=datetime.fromtimestamp(post.created_utc),
            score=post.score,
            upvote_ratio=post.upvote_ratio,
            num_comments=post.num_comments,
            permalink=f"https://www.reddit.com{post.permalink}",
            url=post.url,
            is_self=post.is_self,
            selftext=post.selftext if post.is_self else "",
            subreddit=post.subreddit.display_name,
            link_flair_text=post.link_flair_text,
            category=category,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for database storage or API response.

        Returns:
            Dictionary representation
        """
        return {
            "post_id": self.post_id,
            "title": self.title,
            "author": self.author,
            "created_utc": self.created_utc,
            "score": self.score,
            "upvote_ratio": self.upvote_ratio,
            "num_comments": self.num_comments,
            "permalink": self.permalink,
            "url": self.url,
            "is_self": self.is_self,
            "selftext": self.selftext,
            "cleaned_selftext": self.cleaned_selftext,
            "topic_text": self.topic_text,
            "subreddit": self.subreddit,
            "link_flair_text": self.link_flair_text,
            "category": self.category,
            "embedding": self.embedding,
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "topic_description": self.topic_description,
            "stance_dist_weighted": self.stance_dist_weighted,
            "sentiment_dist_weighted": self.sentiment_dist_weighted,
            "polarization_score": self.polarization_score,
            "snapshot_week": self.snapshot_week,
            "comments": self.comments,
            "photo_parse": self.photo_parse,
            "historical_metrics": self.historical_metrics,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedditPost":
        """
        Create a RedditPost from a dictionary (e.g., from database).

        Args:
            data: Dictionary with post data

        Returns:
            RedditPost instance
        """
        return cls(
            post_id=data.get("post_id", ""),
            title=data.get("title", ""),
            author=data.get("author", ""),
            created_utc=data.get("created_utc", datetime.utcnow()),
            score=data.get("score", 0),
            upvote_ratio=data.get("upvote_ratio", 0.0),
            num_comments=data.get("num_comments", 0),
            permalink=data.get("permalink", ""),
            url=data.get("url", ""),
            is_self=data.get("is_self", False),
            selftext=data.get("selftext", ""),
            cleaned_selftext=data.get("cleaned_selftext", ""),
            topic_text=data.get("topic_text", ""),
            subreddit=data.get("subreddit", ""),
            link_flair_text=data.get("link_flair_text"),
            category=data.get("category", "general"),
            embedding=data.get("embedding"),
            topic_id=data.get("topic_id"),
            topic_name=data.get("topic_name"),
            topic_description=data.get("topic_description"),
            stance_dist_weighted=data.get("stance_dist_weighted"),
            sentiment_dist_weighted=data.get("sentiment_dist_weighted"),
            polarization_score=data.get("polarization_score"),
            snapshot_week=data.get("snapshot_week"),
            comments=data.get("comments", []),
            photo_parse=data.get("photo_parse"),
            historical_metrics=data.get("historical_metrics", []),
            last_updated=data.get("last_updated"),
        )

    def should_fetch_comments(self, min_selftext_length: int = 100) -> bool:
        """
        Determine if comments should be fetched for this post (smart mode logic).

        Args:
            min_selftext_length: Minimum selftext length to skip comments

        Returns:
            True if comments should be fetched, False otherwise
        """
        if not self.is_self:
            # Link/image/video posts - fetch comments
            return True
        elif len(self.selftext.strip()) < min_selftext_length:
            # Short text - fetch comments
            return True
        else:
            # Sufficient text - skip comments
            return False
