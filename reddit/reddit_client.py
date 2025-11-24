"""
Reddit API Client

This module provides a unified interface for interacting with the Reddit API using PRAW.
Centralizes Reddit API initialization and basic API calls.
"""

import os
import logging
from typing import Any, Mapping, Optional

import praw
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedditClient:
    """Unified client for Reddit API interactions using PRAW."""

    def __init__(
        self,
        *,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
    ):
        """Initialize the Reddit API client using provided credentials or fallbacks."""
        config_creds = (config or {}).get("reddit", {}) if config else {}
        client_id = client_id or config_creds.get("client_id") or os.getenv("REDDIT_CLIENT_ID")
        client_secret = client_secret or config_creds.get("client_secret") or os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = user_agent or config_creds.get("user_agent") or os.getenv("REDDIT_USER_AGENT")

        if not all([client_id, client_secret, user_agent]):
            raise ValueError("Reddit API credentials not found in environment variables or config settings.")

        # Initialize PRAW Reddit instance
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

        logger.info("Reddit API client initialized")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RedditClient":
        """Convenience constructor that prefers credentials defined in the YAML settings."""
        return cls(config=config)

    def get_subreddit(self, subreddit_name: str):
        """
        Get a subreddit instance.

        Args:
            subreddit_name: Name of the subreddit

        Returns:
            PRAW Subreddit object
        """
        return self.reddit.subreddit(subreddit_name)

    def get_submission(self, post_id: str):
        """
        Get a submission (post) by ID.

        Args:
            post_id: Reddit post ID

        Returns:
            PRAW Submission object
        """
        return self.reddit.submission(id=post_id)

    def get_top_posts(self, subreddit_name: str, time_filter: str = "week", limit: int = 30):
        """
        Get top posts from a subreddit.

        Args:
            subreddit_name: Name of the subreddit
            time_filter: Time filter (hour, day, week, month, year, all)
            limit: Maximum number of posts to fetch

        Returns:
            PRAW ListingGenerator of submissions
        """
        subreddit = self.get_subreddit(subreddit_name)
        return subreddit.top(time_filter=time_filter, limit=limit)

    def get_hot_posts(self, subreddit_name: str, limit: int = 30):
        """
        Get hot posts from a subreddit.

        Args:
            subreddit_name: Name of the subreddit
            limit: Maximum number of posts to fetch

        Returns:
            PRAW ListingGenerator of submissions
        """
        subreddit = self.get_subreddit(subreddit_name)
        return subreddit.hot(limit=limit)

    def get_new_posts(self, subreddit_name: str, limit: int = 30):
        """
        Get new posts from a subreddit.

        Args:
            subreddit_name: Name of the subreddit
            limit: Maximum number of posts to fetch

        Returns:
            PRAW ListingGenerator of submissions
        """
        subreddit = self.get_subreddit(subreddit_name)
        return subreddit.new(limit=limit)
