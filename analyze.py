"""Trend extraction, clustering, scoring, and report generation.

Non-LLM baseline:
  1. Extract tech entities from each HIGH_SIGNAL post using TECH_VOCABULARY
  2. Build entity → [posts] index
  3. Score clusters with the spec's weighted formula
  4. Generate structured TrendInsight records with evidence quotes
  5. Emit JSON and Markdown reports

The summarization layer is pluggable: swap out TrendAnalyzer._generate_narrative()
with an LLM call when you want richer descriptions.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings
from pipeline import EnrichedPost, PipelineResult
from tech_classifier import TECH_VOCABULARY, SignalType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Display name map for known tech terms
# ---------------------------------------------------------------------------

_DISPLAY: dict[str, str] = {
    "aws": "AWS", "gcp": "GCP", "api": "API", "sdk": "SDK",
    "llm": "LLM", "llms": "LLMs", "rag": "RAG", "ml": "ML", "ai": "AI",
    "ci/cd": "CI/CD", "grpc": "gRPC", "graphql": "GraphQL",
    "k8s": "Kubernetes", "nextjs": "Next.js", "next.js": "Next.js",
    "nuxt": "Nuxt.js", "wasm": "WebAssembly", "webassembly": "WebAssembly",
    "ebpf": "eBPF", "openai": "OpenAI", "anthropic": "Anthropic",
    "hugging face": "Hugging Face", "langchain": "LangChain",
    "llamaindex": "LlamaIndex", "pytorch": "PyTorch",
    "tensorflow": "TensorFlow", "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL", "mongodb": "MongoDB", "redis": "Redis",
    "kafka": "Kafka", "rabbitmq": "RabbitMQ",
    "elasticsearch": "Elasticsearch", "opensearch": "OpenSearch",
    "kubernetes": "Kubernetes", "docker": "Docker", "terraform": "Terraform",
    "ansible": "Ansible", "prometheus": "Prometheus", "grafana": "Grafana",
    "datadog": "DataDog", "github actions": "GitHub Actions",
    "gitlab ci": "GitLab CI", "cloudflare": "Cloudflare", "vercel": "Vercel",
    "typescript": "TypeScript", "javascript": "JavaScript", "python": "Python",
    "golang": "Go", "go": "Go", "rust": "Rust", "react": "React",
    "vue": "Vue", "svelte": "Svelte", "angular": "Angular",
    "django": "Django", "fastapi": "FastAPI", "flask": "Flask",
    "supabase": "Supabase", "planetscale": "PlanetScale", "neon": "Neon",
    "cockroachdb": "CockroachDB", "clickhouse": "ClickHouse",
    "pinecone": "Pinecone", "weaviate": "Weaviate", "ollama": "Ollama",
    "vllm": "vLLM", "opentelemetry": "OpenTelemetry", "sentry": "Sentry",
    "jaeger": "Jaeger", "nginx": "Nginx", "caddy": "Caddy",
    "traefik": "Traefik", "protobuf": "Protocol Buffers",
    "fly.io": "Fly.io", "railway": "Railway", "heroku": "Heroku",
    "digitalocean": "DigitalOcean", "hetzner": "Hetzner",
    "kotlin": "Kotlin", "swift": "Swift", "dart": "Dart",
    "zig": "Zig", "scala": "Scala", "haskell": "Haskell",
    "elixir": "Elixir", "erlang": "Erlang", "clojure": "Clojure",
    "ruby": "Ruby", "php": "PHP", "java": "Java", "remix": "Remix",
    "astro": "Astro", "solid": "SolidJS", "htmx": "HTMX",
    "rails": "Rails", "laravel": "Laravel", "express": "Express",
    "fastify": "Fastify", "gin": "Gin", "fiber": "Fiber",
    "actix": "Actix", "axum": "Axum", "spring": "Spring",
    "istio": "Istio", "envoy": "Envoy", "helm": "Helm",
    "pulumi": "Pulumi", "podman": "Podman", "dynamodb": "DynamoDB",
    "cassandra": "Cassandra", "sqs": "AWS SQS", "pubsub": "GCP Pub/Sub",
    "eventbridge": "EventBridge", "nats": "NATS", "pulsar": "Apache Pulsar",
    "bazel": "Bazel", "cmake": "CMake", "nix": "Nix",
    "onnx": "ONNX", "jax": "JAX", "transformers": "Transformers",
    "langraph": "LangGraph", "crewai": "CrewAI", "autogen": "AutoGen",
    "mistral": "Mistral", "llama": "Llama",
    "microservices": "Microservices", "monolith": "Monolith",
    "serverless": "Serverless", "rest": "REST", "websocket": "WebSocket",
    "openapi": "OpenAPI", "swagger": "Swagger", "azure": "Azure",
    "netlify": "Netlify", "render": "Render", "linode": "Linode",
    "circleci": "CircleCI", "jenkins": "Jenkins", "travis": "Travis CI",
    "argo": "Argo", "cpp": "C++", "c++": "C++",
    "fine-tuning": "Fine-tuning", "embeddings": "Embeddings",
    "vector db": "Vector DB", "feature flag": "Feature Flags",
}


def _display_name(entity: str) -> str:
    return _DISPLAY.get(entity, entity.title())


# ---------------------------------------------------------------------------
# Trend name templates (specific, not generic)
# ---------------------------------------------------------------------------

_TREND_NAMES: dict[SignalType, str] = {
    SignalType.ADOPTION: "{entity} Gaining Real-World Adoption",
    SignalType.COMPLAINT: "Developer Frustration with {entity} Growing",
    SignalType.COMPARISON: "{entity} Under Active Comparison with Alternatives",
    SignalType.ANNOUNCEMENT: "{entity} Draws Community Attention After Recent Announcements",
    SignalType.FEATURE_REQUEST: "Developers Pushing for Better {entity} Capabilities",
    SignalType.PERFORMANCE: "{entity} Performance and Scaling Concerns Surfacing",
    SignalType.PRICING: "{entity} Pricing and Vendor Lock-in Under Scrutiny",
    SignalType.GENERAL: "{entity} in Active Technical Discussion",
}

_WHY_IT_MATTERS: dict[SignalType, str] = {
    SignalType.ADOPTION: (
        "Adoption signals in developer communities precede ecosystem growth — "
        "hiring trends, library support, and long-term maintenance decisions follow. "
        "Multiple corroborating posts increase confidence this is a durable shift."
    ),
    SignalType.COMPLAINT: (
        "Recurring pain-point discussions indicate unresolved friction that may drive "
        "tool abandonment, migration waves, or ecosystem forking. High engagement "
        "amplifies the signal's reliability."
    ),
    SignalType.COMPARISON: (
        "Active tool comparisons emerge when teams are actively evaluating alternatives — "
        "a leading indicator of potential ecosystem churn or consolidation. "
        "The specifics of the trade-offs reveal real-world priority shifts."
    ),
    SignalType.ANNOUNCEMENT: (
        "Community reaction to releases shows which features developers actually care about "
        "versus marketing noise. High engagement = real developer interest, not PR."
    ),
    SignalType.FEATURE_REQUEST: (
        "Repeated feature requests expose gaps between current tooling and real-world needs. "
        "These often predict the next wave of competing tools or forks."
    ),
    SignalType.PERFORMANCE: (
        "Performance discussions at this volume suggest the tool is being used at scale "
        "where theoretical limits become real problems. This signals both maturity and growing pains."
    ),
    SignalType.PRICING: (
        "Pricing frustration at scale drives vendor migrations and open-source alternatives. "
        "This signal is particularly strong when combined with complaints about lock-in."
    ),
    SignalType.GENERAL: (
        "Sustained technical discussion indicates active community engagement and ongoing "
        "developer interest in this technology."
    ),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrendCluster:
    entity: str
    signal_type: SignalType
    posts: list[EnrichedPost] = field(default_factory=list)

    @property
    def post_count(self) -> int:
        return len(self.posts)

    @property
    def total_upvotes(self) -> int:
        return sum(ep.post.score for ep in self.posts)

    @property
    def total_comments(self) -> int:
        return sum(ep.post.num_comments for ep in self.posts)

    @property
    def total_comment_upvotes(self) -> int:
        return sum(ep.comment_upvotes for ep in self.posts)

    @property
    def weighted_score(self) -> float:
        return (
            self.total_upvotes * 0.5
            + self.total_comments * 0.3
            + self.total_comment_upvotes * 0.2
        )


@dataclass
class TrendInsight:
    name: str
    entity: str
    signal_type: str
    what_is_happening: str
    why_it_matters: str
    developers_like: list[str]
    developers_dislike: list[str]
    evidence: list[str]
    source_links: list[str]
    confidence: str
    post_count: int
    total_upvotes: int
    total_comments: int
    weighted_score: float


@dataclass
class TrendReport:
    generated_at: datetime
    subreddits: list[str]
    days: int
    total_posts_analyzed: int
    high_signal_posts: int
    trends: list[TrendInsight]


# ---------------------------------------------------------------------------
# Entity extractor
# ---------------------------------------------------------------------------

# Build patterns once — whole-word matching for multi-word and single-word terms
_ENTITY_RE: dict[str, re.Pattern] = {}
for _term in TECH_VOCABULARY:
    _escaped = re.escape(_term)
    # Allow punctuation boundaries for terms like "next.js" or "fly.io"
    _ENTITY_RE[_term] = re.compile(
        rf"(?<![a-zA-Z0-9]){_escaped}(?![a-zA-Z0-9])", re.IGNORECASE
    )


def _extract_entities(ep: EnrichedPost) -> list[str]:
    parts = [ep.post.title, ep.post.selftext]
    parts.extend(c.body for c in ep.comments[:5])
    text = " ".join(p for p in parts if p)
    tl = text.lower()
    return [term for term, pat in _ENTITY_RE.items() if pat.search(tl)]


# ---------------------------------------------------------------------------
# Sentiment helpers (heuristic, no ML)
# ---------------------------------------------------------------------------

_POSITIVE_RE = re.compile(
    r"\b(?:love|great|excellent|fast|simple|clean|solid|reliable|easy|recommend|"
    r"switched\s+to|works\s+well|happy\s+with|impressed|best|worth\s+it|finally)\b",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"\b(?:hate|broken|slow|frustrat|annoying|avoid|pain|nightmare|bloat|"
    r"complex|terrible|awful|buggy|worse|struggling|disappointment|overpriced|"
    r"abandoned|unmaintained|memory\s+leak|regression|broke)\b",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 20]


def _extract_sentiment_quotes(
    ep: EnrichedPost, entity: str, max_each: int = 3
) -> tuple[list[str], list[str]]:
    entity_re = _ENTITY_RE.get(entity)
    likes: list[str] = []
    dislikes: list[str] = []

    sources = [ep.post.selftext] + [c.body for c in ep.comments if c.score >= 2]
    for block in sources:
        if not block:
            continue
        for sentence in _split_sentences(block):
            if len(sentence) > 250 or len(sentence) < 25:
                continue
            near_entity = entity_re is None or entity_re.search(sentence)
            if _POSITIVE_RE.search(sentence) and near_entity:
                likes.append(sentence)
            elif _NEGATIVE_RE.search(sentence) and near_entity:
                dislikes.append(sentence)

    # Deduplicate by first 60 chars
    def _dedup(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for s in items:
            key = s[:60]
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    return _dedup(likes)[:max_each], _dedup(dislikes)[:max_each]


def _extract_evidence(cluster: TrendCluster, max_quotes: int = 4) -> list[str]:
    """Return high-value comment quotes from a cluster, sorted by score."""
    candidates: list[tuple[int, str]] = []
    for ep in cluster.posts:
        for c in ep.comments:
            body = c.body.strip()
            if c.score < 2 or len(body) < 40 or len(body) > 400:
                continue
            if body.startswith("http") or body.count("\n") > 5:
                continue
            candidates.append((c.score, body))

    candidates.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    quotes: list[str] = []
    for _, body in candidates:
        key = body[:60]
        if key in seen:
            continue
        seen.add(key)
        quotes.append(body[:300])
        if len(quotes) >= max_quotes:
            break

    # Fall back to top post titles if no comments
    if not quotes:
        for ep in sorted(cluster.posts, key=lambda e: e.post.score, reverse=True)[:2]:
            quotes.append(f'[Post] "{ep.post.title}"')

    return quotes


# ---------------------------------------------------------------------------
# Narrative generator (template-based; swap for LLM call here)
# ---------------------------------------------------------------------------


def _generate_what_is_happening(cluster: TrendCluster) -> str:
    entity_name = _display_name(cluster.entity)
    subs = sorted({ep.post.subreddit for ep in cluster.posts})
    sub_str = ", ".join(f"r/{s}" for s in subs[:3])
    top_posts = sorted(cluster.posts, key=lambda e: e.post.score, reverse=True)[:2]
    titles = "; ".join(f'"{ep.post.title[:80]}"' for ep in top_posts)

    templates: dict[SignalType, str] = {
        SignalType.ADOPTION: (
            f"{cluster.post_count} recent discussions across {sub_str} document "
            f"teams actively adopting or migrating to {entity_name}. "
            f"Top-voted threads: {titles}."
        ),
        SignalType.COMPLAINT: (
            f"Developers on {sub_str} are surfacing recurring frustrations with "
            f"{entity_name} across {cluster.post_count} posts ({cluster.total_upvotes:,} upvotes). "
            f"Most-discussed thread: {titles}."
        ),
        SignalType.COMPARISON: (
            f"{cluster.post_count} comparison threads across {sub_str} are actively "
            f"benchmarking or weighing {entity_name} against alternatives. "
            f"Example: {titles}."
        ),
        SignalType.ANNOUNCEMENT: (
            f"Recent {entity_name} releases sparked {cluster.post_count} discussions "
            f"across {sub_str} with {cluster.total_upvotes:,} combined upvotes. "
            f"Notable thread: {titles}."
        ),
        SignalType.FEATURE_REQUEST: (
            f"Developers on {sub_str} are requesting specific improvements to {entity_name} "
            f"across {cluster.post_count} posts. Common theme: {titles}."
        ),
        SignalType.PERFORMANCE: (
            f"{cluster.post_count} threads on {sub_str} report {entity_name} performance, "
            f"scaling, or resource issues with {cluster.total_comments:,} total comments. "
            f"Top thread: {titles}."
        ),
        SignalType.PRICING: (
            f"Pricing and vendor lock-in concerns around {entity_name} appear in "
            f"{cluster.post_count} posts across {sub_str}. Sample thread: {titles}."
        ),
        SignalType.GENERAL: (
            f"{cluster.post_count} substantive technical discussions about {entity_name} "
            f"across {sub_str} with {cluster.total_upvotes:,} upvotes. "
            f"Recent example: {titles}."
        ),
    }
    return templates.get(cluster.signal_type, templates[SignalType.GENERAL])


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class TrendAnalyzer:
    def __init__(self, top_n: Optional[int] = None) -> None:
        self._top_n = top_n or settings.top_trends_count

    def analyze(self, result: PipelineResult) -> TrendReport:
        posts = result.high_signal_posts
        if not posts:
            logger.warning("No high-signal posts — returning empty report")
            return TrendReport(
                generated_at=result.fetched_at,
                subreddits=result.subreddits,
                days=result.days,
                total_posts_analyzed=result.total_posts_fetched,
                high_signal_posts=0,
                trends=[],
            )

        # Build entity → posts index
        entity_posts: dict[str, list[EnrichedPost]] = defaultdict(list)
        for ep in posts:
            for entity in _extract_entities(ep):
                entity_posts[entity].append(ep)

        # Build clusters (minimum 2 posts to surface as a trend)
        clusters: list[TrendCluster] = []
        for entity, eps in entity_posts.items():
            if len(eps) < 2:
                continue
            dominant = self._dominant_type(eps)
            clusters.append(TrendCluster(entity=entity, signal_type=dominant, posts=eps))

        clusters.sort(key=lambda c: c.weighted_score, reverse=True)

        # De-duplicate: one cluster per entity; pick the highest-scored one
        seen: set[str] = set()
        trends: list[TrendInsight] = []
        for cluster in clusters:
            if len(trends) >= self._top_n:
                break
            if cluster.entity in seen:
                continue
            seen.add(cluster.entity)
            trends.append(self._to_insight(cluster))

        logger.info("Identified %d trends from %d high-signal posts", len(trends), len(posts))
        return TrendReport(
            generated_at=result.fetched_at,
            subreddits=result.subreddits,
            days=result.days,
            total_posts_analyzed=result.total_posts_fetched,
            high_signal_posts=len(posts),
            trends=trends,
        )

    # ------------------------------------------------------------------

    def _dominant_type(self, eps: list[EnrichedPost]) -> SignalType:
        counts: dict[SignalType, int] = defaultdict(int)
        for ep in eps:
            counts[ep.classification.signal_type] += 1
        return max(counts, key=counts.get)

    def _to_insight(self, cluster: TrendCluster) -> TrendInsight:
        entity_name = _display_name(cluster.entity)
        name = _TREND_NAMES.get(cluster.signal_type, _TREND_NAMES[SignalType.GENERAL]).format(
            entity=entity_name
        )

        likes, dislikes = [], []
        for ep in cluster.posts:
            l, d = _extract_sentiment_quotes(ep, cluster.entity)
            likes.extend(l)
            dislikes.extend(d)
        likes = list(dict.fromkeys(likes))[:3]
        dislikes = list(dict.fromkeys(dislikes))[:3]

        evidence = _extract_evidence(cluster)
        sources = list(
            dict.fromkeys(
                ep.post.permalink
                for ep in sorted(cluster.posts, key=lambda e: e.post.score, reverse=True)[:5]
            )
        )

        if cluster.post_count >= 5 and cluster.weighted_score >= 100:
            confidence = "high"
        elif cluster.post_count >= 3 and cluster.weighted_score >= 30:
            confidence = "medium"
        else:
            confidence = "low"

        return TrendInsight(
            name=name,
            entity=entity_name,
            signal_type=cluster.signal_type.value,
            what_is_happening=_generate_what_is_happening(cluster),
            why_it_matters=_WHY_IT_MATTERS.get(cluster.signal_type, _WHY_IT_MATTERS[SignalType.GENERAL]),
            developers_like=likes,
            developers_dislike=dislikes,
            evidence=evidence,
            source_links=sources,
            confidence=confidence,
            post_count=cluster.post_count,
            total_upvotes=cluster.total_upvotes,
            total_comments=cluster.total_comments,
            weighted_score=round(cluster.weighted_score, 1),
        )

    # ------------------------------------------------------------------
    # Report generators
    # ------------------------------------------------------------------

    def to_json(self, report: TrendReport) -> str:
        def _serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Not serializable: {type(obj)}")

        data = {
            "generated_at": report.generated_at.isoformat(),
            "subreddits": report.subreddits,
            "days": report.days,
            "total_posts_analyzed": report.total_posts_analyzed,
            "high_signal_posts": report.high_signal_posts,
            "trends": [
                {
                    "name": t.name,
                    "entity": t.entity,
                    "signal_type": t.signal_type,
                    "confidence": t.confidence,
                    "post_count": t.post_count,
                    "total_upvotes": t.total_upvotes,
                    "total_comments": t.total_comments,
                    "weighted_score": t.weighted_score,
                    "what_is_happening": t.what_is_happening,
                    "why_it_matters": t.why_it_matters,
                    "developers_like": t.developers_like,
                    "developers_dislike": t.developers_dislike,
                    "evidence": t.evidence,
                    "source_links": t.source_links,
                }
                for t in report.trends
            ],
        }
        return json.dumps(data, indent=2, default=_serial)

    def to_markdown(self, report: TrendReport) -> str:
        lines: list[str] = []
        sub_str = ", ".join(f"r/{s}" for s in report.subreddits)
        lines.append("# Reddit Tech Trend Report")
        lines.append("")
        lines.append(
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}  "
        )
        lines.append(f"**Subreddits:** {sub_str}  ")
        lines.append(f"**Period:** last {report.days} day(s)")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Posts fetched | {report.total_posts_analyzed:,} |")
        lines.append(f"| High-signal posts | {report.high_signal_posts:,} |")
        lines.append(f"| Trends identified | {len(report.trends)} |")
        lines.append("")

        if not report.trends:
            lines.append("_No trends identified — try a wider time window or more subreddits._")
            return "\n".join(lines)

        lines.append("---")
        lines.append("")

        for i, t in enumerate(report.trends, 1):
            lines.append(f"## Trend {i}: {t.name}")
            lines.append("")
            lines.append(
                f"**Signal:** `{t.signal_type}` | "
                f"**Confidence:** `{t.confidence}` | "
                f"**Posts:** {t.post_count} | "
                f"**Upvotes:** {t.total_upvotes:,} | "
                f"**Comments:** {t.total_comments:,} | "
                f"**Score:** {t.weighted_score:.1f}"
            )
            lines.append("")

            lines.append("### What is happening")
            lines.append("")
            lines.append(t.what_is_happening)
            lines.append("")

            lines.append("### Why it matters")
            lines.append("")
            lines.append(t.why_it_matters)
            lines.append("")

            if t.developers_like:
                lines.append("### What developers like")
                lines.append("")
                for item in t.developers_like:
                    lines.append(f'- "{item}"')
                lines.append("")

            if t.developers_dislike:
                lines.append("### What developers dislike")
                lines.append("")
                for item in t.developers_dislike:
                    lines.append(f'- "{item}"')
                lines.append("")

            if t.evidence:
                lines.append("### Evidence")
                lines.append("")
                for quote in t.evidence:
                    wrapped = textwrap.fill(quote, width=100)
                    lines.append(f"> {wrapped.replace(chr(10), chr(10) + '> ')}")
                    lines.append("")

            if t.source_links:
                lines.append("### Sources")
                lines.append("")
                for link in t.source_links:
                    lines.append(f"- {link}")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def save_reports(
        self, report: TrendReport, output_dir: Optional[Path] = None
    ) -> tuple[Path, Path]:
        out = output_dir or settings.output_dir
        out.mkdir(parents=True, exist_ok=True)
        ts = report.generated_at.strftime("%Y%m%d_%H%M%S")
        json_path = out / f"trends_{ts}.json"
        md_path = out / f"trends_{ts}.md"
        json_path.write_text(self.to_json(report))
        md_path.write_text(self.to_markdown(report))
        logger.info("Reports saved → %s | %s", json_path, md_path)
        return json_path, md_path
