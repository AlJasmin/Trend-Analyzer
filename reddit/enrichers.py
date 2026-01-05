"""
Reddit data enrichers.

This module contains classes for enriching Reddit posts with additional data.
"""
from typing import Optional, Dict, Any
import logging
from .models import RedditPost
from .fetchers import CommentFetcher

logger = logging.getLogger(__name__)


class ImageEnricher:
    """Enriches posts with image analysis."""

    def __init__(self, db_client=None, embedder_model='clip'):
        from processing.no_meme_VLM import ImageEmbedder
        from modeling.image_topic_model import ImageTopicModeler
        from database.image_analysis import ImageAnalysisDB
        self.embedder = ImageEmbedder(model_name=embedder_model)
        self.model_id = getattr(self.embedder, "model_id", None) or getattr(self.embedder, "model_name", "clip")
        self.topic_modeler = ImageTopicModeler()
        self.db = db_client or ImageAnalysisDB()
        self.stats = {"analyzed": 0, "cached": 0}
        self._processed_urls = set()

    def _already_processed(self, url: str) -> bool:
        if not url:
            return False
        if url in self._processed_urls:
            return True
        if hasattr(self.db, "has_embedding") and self.db.has_embedding(url, model=self.model_id):
            self._processed_urls.add(url)
            return True
        return False

    def _mark_processed(self, url: str) -> None:
        if url:
            self._processed_urls.add(url)

    def enrich_post(
        self,
        post: RedditPost,
        existing_post: Optional[Dict] = None
    ) -> RedditPost:
        """
        Enrich a post with image embedding and topic assignment.
        """
        # Only process image posts
        if not hasattr(post, 'url') or not post.url.lower().endswith(('.jpg', '.png', '.gif', '.jpeg', '.webp')):
            return post

        # Check cache first
        if self._already_processed(post.url):
            self.stats["cached"] += 1
            return post

        try:
            embedding = self.embedder.get_embedding(post.url)
            # For batch topic assignment, collect all embeddings and run topic model periodically
            topic = None  # Topic assignment deferred to batch
            self.db.save_embedding(
                post_id=post.post_id,
                image_url=post.url,
                embedding=embedding,
                topic=topic,
                timestamp=post.created_utc,
                model=self.model_id,
            )
            post.photo_parse = embedding  # Optionally store embedding in post
            self.stats["analyzed"] += 1
            self._mark_processed(post.url)
        except Exception as e:
            logger.error(f"Error embedding image for post {post.post_id}: {e}")
        return post

    def get_stats(self) -> dict:
        """Get image analysis statistics."""
        return self.stats.copy()


class CommentEnricher:
    """Enriches posts with comment data."""

    def __init__(self, comment_fetcher: CommentFetcher):
        self.comment_fetcher = comment_fetcher

    def enrich_post(
        self,
        post: RedditPost,
        fetch_mode: str,
        limit: Optional[int] = None
    ) -> RedditPost:
        """
        Enrich a post with comments based on fetch mode.

        Args:
            post: RedditPost object to enrich
            fetch_mode: Comment fetch mode ('true', 'false', 'smart')
            limit: Maximum number of comments to fetch

        Returns:
            Enriched RedditPost object
        """
        if fetch_mode == "false":
            return post

        if fetch_mode == "smart" and post.score < 100:
            return post

        try:
            comments = self.comment_fetcher.fetch_top_comments(post.post_id, limit)
            post.comments = comments
        except Exception as e:
            logger.error(f"Error enriching comments for post {post.post_id}: {e}")

        return post

    def get_stats(self) -> dict:
        """Get comment enrichment statistics."""
        return self.comment_fetcher.get_stats()
