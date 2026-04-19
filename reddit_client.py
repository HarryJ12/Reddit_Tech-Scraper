"""Reddit API client with token-bucket rate limiting, file caching, and exponential backoff."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class RedditPost(BaseModel):
    post_id: str
    title: str
    selftext: str
    score: int
    num_comments: int
    created_utc: datetime
    permalink: str
    subreddit: str
    url: str
    author: str


class RedditComment(BaseModel):
    comment_id: str
    post_id: str
    body: str
    score: int
    created_utc: datetime
    author: str


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Single-threaded token bucket rate limiter."""

    def __init__(self, requests_per_minute: int) -> None:
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._capacity = float(requests_per_minute)
        self._tokens = self._capacity
        self._last = time.monotonic()

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._last) * self._rate,
            )
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
            logger.debug("Rate limiter sleeping %.2fs", wait)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# File-based cache with TTL
# ---------------------------------------------------------------------------


class _FileCache:
    def __init__(self, directory: Path, ttl_hours: int) -> None:
        self._dir = directory
        self._ttl = ttl_hours * 3600
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._dir / f"{digest}.json"

    def get(self, key: str) -> Optional[dict | list]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            record = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if time.time() - record["ts"] > self._ttl:
            p.unlink(missing_ok=True)
            return None
        return record["data"]

    def put(self, key: str, data: dict | list) -> None:
        p = self._path(key)
        try:
            p.write_text(json.dumps({"ts": time.time(), "data": data}))
        except OSError as exc:
            logger.warning("Cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# Reddit client
# ---------------------------------------------------------------------------


class RedditClient:
    _OAUTH_BASE = "https://oauth.reddit.com"
    _PUBLIC_BASE = "https://www.reddit.com"
    _TOKEN_URL = "https://www.reddit.com/api/v1/access_token"

    def __init__(self) -> None:
        self._limiter = _TokenBucket(settings.requests_per_minute)
        self._cache = _FileCache(settings.cache_dir, settings.cache_ttl_hours)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = settings.reddit_user_agent
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._seen_posts: set[str] = set()
        self._seen_comments: set[str] = set()
        self._use_oauth: bool = bool(
            settings.reddit_client_id and settings.reddit_client_secret
        )
        if not self._use_oauth:
            logger.warning(
                "No OAuth credentials — using public JSON endpoints. "
                "Set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET in .env for better limits."
            )

    # ------------------------------------------------------------------
    # OAuth
    # ------------------------------------------------------------------

    def _refresh_token(self) -> str:
        resp = self._session.post(
            self._TOKEN_URL,
            auth=(settings.reddit_client_id, settings.reddit_client_secret),
            data={"grant_type": "client_credentials"},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expiry = time.time() + payload.get("expires_in", 3600) - 60
        logger.info("OAuth token refreshed (expires in %ds)", payload.get("expires_in", 3600))
        return self._token

    def _get_token(self) -> str:
        if not self._token or time.time() >= self._token_expiry:
            return self._refresh_token()
        return self._token

    # ------------------------------------------------------------------
    # Core HTTP with caching and backoff
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> dict | list:
        if self._use_oauth:
            url = f"{self._OAUTH_BASE}{path}"
        else:
            url = f"{self._PUBLIC_BASE}{path}.json"

        cache_key = url + json.dumps(params or {}, sort_keys=True)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit: %s", url)
            return cached

        extra_headers: dict[str, str] = {}
        if self._use_oauth:
            extra_headers["Authorization"] = f"bearer {self._get_token()}"

        for attempt in range(settings.retry_max_attempts):
            self._limiter.acquire()
            try:
                resp = self._session.get(
                    url, params=params, headers=extra_headers, timeout=20
                )
                if resp.status_code == 429:
                    # Hard stop on rate limit — respect platform rules
                    retry_after = int(resp.headers.get("Retry-After", 120))
                    logger.warning(
                        "HTTP 429 — backing off %ds before retrying", retry_after
                    )
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                data = resp.json()
                self._cache.put(cache_key, data)
                return data
            except requests.RequestException as exc:
                if attempt == settings.retry_max_attempts - 1:
                    raise
                delay = settings.retry_base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "Request error (%s) — retry %d/%d in %.1fs",
                    exc,
                    attempt + 1,
                    settings.retry_max_attempts,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(f"All retries exhausted for {url}")

    def get_url(self, url: str, params: Optional[dict] = None) -> dict | list:
        """Fetch an arbitrary Reddit JSON URL (used for --urls CLI flag)."""
        cache_key = url + json.dumps(params or {}, sort_keys=True)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit: %s", url)
            return cached

        for attempt in range(settings.retry_max_attempts):
            self._limiter.acquire()
            try:
                resp = self._session.get(
                    url, params=params, headers={}, timeout=20
                )
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 120))
                    logger.warning("HTTP 429 — backing off %ds", retry_after)
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                data = resp.json()
                self._cache.put(cache_key, data)
                return data
            except requests.RequestException as exc:
                if attempt == settings.retry_max_attempts - 1:
                    raise
                delay = settings.retry_base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning("Request error (%s) — retry %d in %.1fs", exc, attempt + 1, delay)
                time.sleep(delay)

        raise RuntimeError(f"All retries exhausted for {url}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_posts(
        self,
        subreddit: str,
        sort: str = "hot",
        time_filter: str = "week",
        limit: int = 50,
        days: int = 7,
    ) -> list[RedditPost]:
        path = f"/r/{subreddit}/{sort}"
        params: dict = {
            "limit": min(limit, 100),
            "t": time_filter,
            "raw_json": "1",
        }
        cutoff_ts = datetime.now(timezone.utc).timestamp() - days * 86400
        posts: list[RedditPost] = []
        after: Optional[str] = None

        while len(posts) < limit:
            if after:
                params["after"] = after
            try:
                data = self._get(path, params)
            except Exception as exc:
                logger.error("Failed to fetch r/%s page: %s", subreddit, exc)
                break

            listing = data.get("data", {}) if isinstance(data, dict) else {}
            children = listing.get("children", [])
            if not children:
                break

            for child in children:
                if len(posts) >= limit:
                    break
                pd = child.get("data", {})

                if pd.get("stickied") or pd.get("distinguished") == "moderator":
                    continue
                created = pd.get("created_utc", 0)
                if created < cutoff_ts:
                    continue
                pid = pd.get("id", "")
                if not pid or pid in self._seen_posts:
                    continue
                self._seen_posts.add(pid)

                posts.append(
                    RedditPost(
                        post_id=pid,
                        title=pd.get("title", ""),
                        selftext=(pd.get("selftext") or ""),
                        score=pd.get("score", 0),
                        num_comments=pd.get("num_comments", 0),
                        created_utc=datetime.fromtimestamp(created, tz=timezone.utc),
                        permalink=f"https://reddit.com{pd.get('permalink', '')}",
                        subreddit=pd.get("subreddit", subreddit),
                        url=pd.get("url", ""),
                        author=str(pd.get("author", "")),
                    )
                )

            after = listing.get("after")
            if not after:
                break

        logger.info("Fetched %d posts from r/%s", len(posts), subreddit)
        return posts

    def fetch_posts_from_url(self, url: str, days: int = 7) -> list[RedditPost]:
        """Parse posts from a direct Reddit listing URL (e.g. .../new/.json)."""
        cutoff_ts = datetime.now(timezone.utc).timestamp() - days * 86400
        try:
            data = self.get_url(url, params={"raw_json": "1", "limit": "100"})
        except Exception as exc:
            logger.error("Failed to fetch URL %s: %s", url, exc)
            return []

        listing = data.get("data", {}) if isinstance(data, dict) else {}
        children = listing.get("children", [])
        posts: list[RedditPost] = []
        for child in children:
            pd = child.get("data", {})
            if pd.get("stickied"):
                continue
            created = pd.get("created_utc", 0)
            if created < cutoff_ts:
                continue
            pid = pd.get("id", "")
            if not pid or pid in self._seen_posts:
                continue
            self._seen_posts.add(pid)
            subreddit = pd.get("subreddit", "unknown")
            posts.append(
                RedditPost(
                    post_id=pid,
                    title=pd.get("title", ""),
                    selftext=(pd.get("selftext") or ""),
                    score=pd.get("score", 0),
                    num_comments=pd.get("num_comments", 0),
                    created_utc=datetime.fromtimestamp(created, tz=timezone.utc),
                    permalink=f"https://reddit.com{pd.get('permalink', '')}",
                    subreddit=subreddit,
                    url=pd.get("url", ""),
                    author=str(pd.get("author", "")),
                )
            )
        logger.info("Fetched %d posts from URL %s", len(posts), url)
        return posts

    def fetch_comments(
        self, post: RedditPost, max_comments: int = 20
    ) -> list[RedditComment]:
        path = f"/r/{post.subreddit}/comments/{post.post_id}"
        params = {"depth": "1", "limit": str(max_comments), "raw_json": "1"}
        try:
            data = self._get(path, params)
        except Exception as exc:
            logger.warning("Comments fetch failed for %s: %s", post.post_id, exc)
            return []

        if not isinstance(data, list) or len(data) < 2:
            return []

        comments: list[RedditComment] = []
        for child in data[1].get("data", {}).get("children", []):
            if child.get("kind") != "t1":
                continue
            cd = child.get("data", {})
            body = cd.get("body", "")
            if not body or body in ("[deleted]", "[removed]"):
                continue
            cid = cd.get("id", "")
            if not cid or cid in self._seen_comments:
                continue
            self._seen_comments.add(cid)
            created = cd.get("created_utc", 0)
            comments.append(
                RedditComment(
                    comment_id=cid,
                    post_id=post.post_id,
                    body=body,
                    score=cd.get("score", 0),
                    created_utc=datetime.fromtimestamp(created, tz=timezone.utc),
                    author=str(cd.get("author", "")),
                )
            )
        return comments[:max_comments]
