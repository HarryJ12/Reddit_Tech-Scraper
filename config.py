"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class _Settings:
    # ------------------------------------------------------------------
    # Reddit API
    # ------------------------------------------------------------------
    reddit_client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    reddit_user_agent: str = os.getenv(
        "REDDIT_USER_AGENT",
        "tech-trend-monitor/1.0 (educational research; contact: change-me@example.com)",
    )

    # ------------------------------------------------------------------
    # Rate limiting
    # Authenticated (OAuth):   up to 60 req/min — we use 45 for safety
    # Public JSON endpoints:   ~1 req/sec — we use 20/min (conservative)
    # ------------------------------------------------------------------
    requests_per_minute: int = int(os.getenv("REQUESTS_PER_MINUTE", "30"))
    retry_max_attempts: int = 3
    retry_base_delay: float = 2.0  # seconds; doubles each retry

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    cache_dir: Path = Path(os.getenv("CACHE_DIR", ".cache"))
    cache_ttl_hours: int = int(os.getenv("CACHE_TTL_HOURS", "5"))

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    top_trends_count: int = int(os.getenv("TOP_TRENDS_COUNT", "10"))
    min_post_score: int = int(os.getenv("MIN_POST_SCORE", "300"))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "output"))

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    topic_subreddits: dict[str, list[str]] = {
        "ai": [
            "MachineLearning", "artificial", "LanguageTechnology", "deeplearning",
            "LLMDevs", "computervision", "reinforcementlearning", "LocalLLaMA",
            "MLQuestions", "datascience",
        ],
        "ml": [
            "MachineLearning", "deeplearning", "LocalLLaMA", "datascience",
            "LanguageTechnology", "computervision", "reinforcementlearning", "LLMDevs",
            "MLQuestions", "learnmachinelearning",
        ],
        "genai": [
            "LocalLLaMA", "StableDiffusion", "Midjourney", "deeplearning",
            "LLMDevs", "genai", "generativeAI", "LanguageModels",
            "computervision", "aivideo",
        ],
        "programming": [
            "programming", "webdev", "devops", "SoftwareEngineering",
            "cpp", "compsci", "ExperiencedDevs",
        ],
        "cloud": [
            "aws", "devops", "kubernetes", "docker", "cloudcomputing",
            "googlecloud", "azure", "Terraform", "sysadmin", "learncloud",
        ],
        "startups": [
            "startups", "SaaS", "indiehackers", "SideProject", "Entrepreneur",
            "alphaandbetausers", "EntrepreneurRideAlong", "MicroSaaS",
            "SaaSMarketing", "leanstartup",
        ],
        "tech": [
            "technology", "programming", "devops", "MachineLearning", "compsci",
            "hardware", "electronics", "informationtechnology", "tech", "netsec",
        ],
        "news": [
            "technology", "programming", "MachineLearning", "devops", "startups",
            "news", "science", "tech", "arsTechnica", "Wired",
        ],
        "robotics": [
            "robotics", "ROS", "embedded", "humanoidrobots", "drone",
            "electronics", "automation", "MechanicalEngineering", "computervision",
            "reinforcementlearning",
        ],
        "quantum": [
            "QuantumComputing", "QuantumInformation", "compsci", "MachineLearning",
            "physics", "QuantumPhysics", "science", "quantum",
        ],
        "security": [
            "cybersecurity", "netsec", "blueteamsec", "redteamsec", "ethicalhacking",
            "AskNetsec", "Malware", "ReverseEngineering", "InfoSecNews", "hacking",
        ],
    }

    default_subreddits: list[str] = topic_subreddits["ai"]


settings = _Settings()
