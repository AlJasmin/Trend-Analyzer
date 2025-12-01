"""
Reddit post filters.

This module contains classes for filtering and sorting Reddit posts.
"""
import logging
from datetime import datetime, timedelta
from typing import List
from .models import RedditPost

class PostFilter:
    """Filters and processes Reddit posts based on various criteria."""

    def __init__(self, logger_instance=None):
        """Initialize the post filter."""
        self.logger = logger_instance or logging.getLogger(__name__)

    def filter_by_score(self, posts: List[RedditPost], min_score: int = 0) -> List[RedditPost]:
        """
        Filter posts by minimum score.

        Args:
            posts: List of RedditPost objects
            min_score: Minimum score threshold

        Returns:
            Filtered list of posts
        """
        filtered = [post for post in posts if post.score >= min_score]
        self.logger.info("Filtered %s posts to %s with score >= %s", len(posts), len(filtered), min_score)
        return filtered

    def filter_by_recency(self, posts: List[RedditPost], days: int = 7) -> List[RedditPost]:
        """
        Filter posts by recency.

        Args:
            posts: List of RedditPost objects
            days: Number of days to look back

        Returns:
            Filtered list of posts
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        filtered = [post for post in posts if post.created_utc >= cutoff]
        self.logger.info("Filtered %s posts to %s from last %s days", len(posts), len(filtered), days)
        return filtered

    def filter_by_category(self, posts: List[RedditPost], categories: List[str]) -> List[RedditPost]:
        """
        Filter posts by category.

        Args:
            posts: List of RedditPost objects
            categories: List of allowed categories

        Returns:
            Filtered list of posts
        """
        filtered = [post for post in posts if post.category in categories]
        self.logger.info("Filtered %s posts to %s with categories %s", len(posts), len(filtered), categories)
        return filtered

    def exclude_by_category(self, posts: List[RedditPost], excluded_categories: List[str]) -> List[RedditPost]:
        """
        Exclude posts by category.

        Args:
            posts: List of RedditPost objects
            excluded_categories: List of categories to exclude

        Returns:
            Filtered list of posts
        """
        filtered = [post for post in posts if post.category not in excluded_categories]
        self.logger.info("Filtered %s posts to %s excluding %s", len(posts), len(filtered), excluded_categories)
        return filtered

    def deduplicate(self, posts: List[RedditPost]) -> List[RedditPost]:
        """
        Remove duplicate posts based on post_id.

        Args:
            posts: List of RedditPost objects

        Returns:
            Deduplicated list of posts
        """
        seen_ids = set()
        unique_posts = []

        for post in posts:
            if post.post_id not in seen_ids:
                seen_ids.add(post.post_id)
                unique_posts.append(post)

        if len(posts) != len(unique_posts):
            self.logger.info("Removed %s duplicate posts", len(posts) - len(unique_posts))

        return unique_posts

    def sort_by_score(self, posts: List[RedditPost], descending: bool = True) -> List[RedditPost]:
        """
        Sort posts by score.

        Args:
            posts: List of RedditPost objects
            descending: Sort in descending order (default: True)

        Returns:
            Sorted list of posts
        """
        sorted_posts = sorted(posts, key=lambda p: p.score, reverse=descending)
        self.logger.debug("Sorted %s posts by score (descending=%s)", len(posts), descending)
        return sorted_posts

    def sort_by_recency(self, posts: List[RedditPost], descending: bool = True) -> List[RedditPost]:
        """
        Sort posts by creation time.

        Args:
            posts: List of RedditPost objects
            descending: Sort in descending order (default: True)

        Returns:
            Sorted list of posts
        """
        sorted_posts = sorted(posts, key=lambda p: p.created_utc, reverse=descending)
        self.logger.debug("Sorted %s posts by recency (descending=%s)", len(posts), descending)
        return sorted_posts

    def get_top_n(self, posts: List[RedditPost], n: int) -> List[RedditPost]:
        """
        Get top N posts.

        Args:
            posts: List of RedditPost objects
            n: Number of posts to return

        Returns:
            Top N posts
        """
        return posts[:n]
