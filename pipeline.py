"""Data collection pipeline: fetches, filters, and enriches Reddit posts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings
from reddit_client import RedditClient, RedditComment, RedditPost
from tech_classifier import ClassificationResult, SignalLevel, classify

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass
class EnrichedPost:
    post: RedditPost
    classification: ClassificationResult
    comments: list[RedditComment] = field(default_factory=list)

    @property
    def comment_upvotes(self) -> int:
        return sum(max(c.score, 0) for c in self.comments)

    @property
    def weighted_score(self) -> float:
        """Spec formula: post_score*0.5 + num_comments*0.3 + comment_upvotes*0.2"""
        return (
            self.post.score * 0.5
            + self.post.num_comments * 0.3
            + self.comment_upvotes * 0.2
        )


@dataclass
class PipelineResult:
    subreddits: list[str]
    days: int
    fetched_at: datetime
    total_posts_fetched: int
    high_signal_posts: list[EnrichedPost]
    low_signal_posts: list[EnrichedPost]

    @property
    def all_posts(self) -> list[EnrichedPost]:
        return self.high_signal_posts + self.low_signal_posts


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    def __init__(
        self,
        subreddits: Optional[list[str]] = None,
        days: int = 7,
        limit_per_subreddit: int = 50,
        fetch_comments: bool = True,
        max_comments_per_post: int = 15,
        output_dir: Optional[Path] = None,
        direct_urls: Optional[list[str]] = None,
    ) -> None:
        self.subreddits = subreddits or settings.default_subreddits
        self.days = days
        self.limit = limit_per_subreddit
        self.fetch_comments = fetch_comments
        self.max_comments = max_comments_per_post
        self.output_dir = output_dir or settings.output_dir
        self.direct_urls = direct_urls or []
        self._client = RedditClient()

    def run(self) -> PipelineResult:
        high: list[EnrichedPost] = []
        low: list[EnrichedPost] = []
        total = 0

        # Fetch from named subreddits
        for sub in self.subreddits:
            logger.info("Fetching r/%s (last %d days, limit %d)", sub, self.days, self.limit)
            try:
                posts = self._client.fetch_posts(
                    subreddit=sub,
                    sort="hot",
                    time_filter=self._time_filter(),
                    limit=self.limit,
                    days=self.days,
                )
            except Exception as exc:
                logger.error("Skipping r/%s — %s", sub, exc)
                continue
            total += len(posts)
            self._process_posts(posts, high, low)

        # Fetch from direct URLs
        for url in self.direct_urls:
            logger.info("Fetching direct URL: %s", url)
            try:
                posts = self._client.fetch_posts_from_url(url, days=self.days)
            except Exception as exc:
                logger.error("Skipping URL %s — %s", url, exc)
                continue
            total += len(posts)
            self._process_posts(posts, high, low)

        logger.info(
            "Pipeline complete: %d total, %d high-signal, %d low-signal",
            total, len(high), len(low),
        )
        result = PipelineResult(
            subreddits=self.subreddits,
            days=self.days,
            fetched_at=datetime.now(timezone.utc),
            total_posts_fetched=total,
            high_signal_posts=high,
            low_signal_posts=low,
        )
        self._save_raw(result)
        return result

    def _process_posts(
        self,
        posts: list[RedditPost],
        high: list[EnrichedPost],
        low: list[EnrichedPost],
    ) -> None:
        for post in posts:
            if post.score < settings.min_post_score:
                continue
            clf = classify(post)
            comments: list[RedditComment] = []
            if self.fetch_comments and clf.signal_level == SignalLevel.HIGH:
                try:
                    comments = self._client.fetch_comments(post, self.max_comments)
                except Exception as exc:
                    logger.warning("Skipping comments for %s: %s", post.post_id, exc)

            ep = EnrichedPost(post=post, classification=clf, comments=comments)
            if clf.signal_level == SignalLevel.HIGH:
                high.append(ep)
            else:
                low.append(ep)

    def _time_filter(self) -> str:
        if self.days <= 1:
            return "day"
        if self.days <= 7:
            return "week"
        if self.days <= 30:
            return "month"
        return "year"

    def _save_raw(self, result: PipelineResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = result.fetched_at.strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"raw_{ts}.json"
        records = []
        for ep in result.all_posts:
            records.append(
                {
                    "post": ep.post.model_dump(mode="json"),
                    "signal_level": ep.classification.signal_level,
                    "signal_type": ep.classification.signal_type,
                    "signal_score": ep.classification.signal_score,
                    "tags": ep.classification.matched_tags,
                    "comments": [c.model_dump(mode="json") for c in ep.comments],
                }
            )
        payload = {
            "meta": {
                "subreddits": result.subreddits,
                "days": result.days,
                "fetched_at": result.fetched_at.isoformat(),
                "total_fetched": result.total_posts_fetched,
                "high_signal": len(result.high_signal_posts),
                "low_signal": len(result.low_signal_posts),
            },
            "posts": records,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Raw data saved → %s", path)
