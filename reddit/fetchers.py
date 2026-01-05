"""
Reddit data fetchers.

This module contains classes for fetching different types of Reddit data.
"""
from typing import List, Optional
import logging
import re
from .models import RedditPost
from .reddit_client import RedditClient

logger = logging.getLogger(__name__)

BOT_COMMENT_PATTERNS = [
    r"your post is getting popular",
    r"this post has reached",
    r"i am a bot",
    r"^beep boop",
    r"this action was performed automatically",
    r"^[\*\s]*bot",
    r"contact the moderators",
    r"^automod",
    r"^automoderator",
    r"please contact the moderators",
    r"if you have any questions or concerns",
]


def _is_bot_comment(comment_body: str) -> bool:
    if not comment_body:
        return True
    comment_lower = comment_body.lower().strip()
    for pattern in BOT_COMMENT_PATTERNS:
        if re.search(pattern, comment_lower, re.IGNORECASE):
            return True
    return False


def _filter_bot_comments(comments: List[dict]) -> List[dict]:
    filtered = [c for c in comments if not _is_bot_comment(c.get("body", ""))]
    removed = len(comments) - len(filtered)
    if removed:
        logger.debug("Filtered %s suspected bot comments", removed)
    return filtered

class PostFetcher:
    """Fetches Reddit posts and converts them to RedditPost objects."""

    def __init__(self, client: RedditClient):
        self.client = client

    def fetch_posts(
        self,
        subreddit: str,
        *,
        feed: str = "top",
        time_filter: str = "week",
        limit: int = 100,
    ) -> List[RedditPost]:
        """
        Fetch posts from a subreddit using the selected feed type.

        Args:
            subreddit: Subreddit name
            feed: Feed to use ('top', 'hot', 'new')
            time_filter: Time filter for 'top' feed (hour, day, week, month, year, all)
            limit: Maximum number of posts to fetch

        Returns:
            List of RedditPost objects
        """
        feed = (feed or "top").lower()
        try:
            if feed == "top":
                submissions = self.client.get_top_posts(subreddit, time_filter, limit)
            elif feed == "hot":
                submissions = self.client.get_hot_posts(subreddit, limit)
            elif feed == "new":
                submissions = self.client.get_new_posts(subreddit, limit)
            else:
                raise ValueError(f"Unsupported feed '{feed}'. Use 'top', 'hot', or 'new'.")
            return [RedditPost.from_praw(submission) for submission in submissions]
        except Exception as e:
            logger.error(f"Error fetching {feed} posts from r/{subreddit}: {e}")
            return []

    def fetch_top_posts(
        self,
        subreddit: str,
        time_filter: str = "week",
        limit: int = 100
    ) -> List[RedditPost]:
        """Compatibility wrapper for legacy codepaths."""
        return self.fetch_posts(
            subreddit,
            feed="top",
            time_filter=time_filter,
            limit=limit,
        )


class CommentFetcher:
    """Fetches and processes Reddit comments."""

    def __init__(self, client: RedditClient):
        self.client = client
        self.stats = {"fetched": 0, "skipped": 0}

    def fetch_top_comments(
        self,
        post_id: str,
        limit: Optional[int] = None,
        min_count: Optional[int] = None,
    ) -> List[dict]:
        """
        Fetch top comments for a post.

        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments to fetch

        Returns:
            List of comment dictionaries
        """
        try:
            submission = self.client.get_submission(post_id)
            comments = []
            
            # Ensure comments are loaded
            submission.comments.replace_more(limit=0)
            
            # Get top-level comments
            comment_iter = submission.comments if limit is None else submission.comments[:limit]
            for comment in comment_iter:
                comment_dict = {
                    'id': comment.id,
                    'post_id': post_id,
                    'author': str(comment.author),
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': comment.created_utc
                }
                comments.append(comment_dict)
                self.stats["fetched"] += 1
            
            comments = _filter_bot_comments(comments)
            if min_count is not None and len(comments) < min_count:
                self.stats["skipped"] += 1
                return []

            return comments
        except Exception as e:
            logger.error(f"Error fetching comments for post {post_id}: {e}")
            self.stats["skipped"] += 1
            return []

    def get_stats(self) -> dict:
        """Get comment fetching statistics."""
        return self.stats.copy()
