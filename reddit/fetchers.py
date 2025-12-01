"""
Reddit data fetchers.

This module contains classes for fetching different types of Reddit data.
"""
from typing import List, Optional
import logging
from .models import RedditPost
from .reddit_client import RedditClient

logger = logging.getLogger(__name__)


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
        limit: Optional[int] = None
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
            for comment in submission.comments[:limit]:
                comment_dict = {
                    'id': comment.id,
                    'author': str(comment.author),
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': comment.created_utc
                }
                comments.append(comment_dict)
                self.stats["fetched"] += 1
            
            return comments
        except Exception as e:
            logger.error(f"Error fetching comments for post {post_id}: {e}")
            self.stats["skipped"] += 1
            return []

    def get_stats(self) -> dict:
        """Get comment fetching statistics."""
        return self.stats.copy()
