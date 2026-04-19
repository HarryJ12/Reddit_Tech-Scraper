"""Signal classifier: tags Reddit posts as HIGH_SIGNAL or LOW_SIGNAL for tech trend analysis.

Design philosophy:
- Heuristics-first (keyword patterns + structural signals)
- Easily replaceable with an ML model by swapping out classify()
- No external dependencies beyond the stdlib
"""

from __future__ import annotations

import re
from enum import Enum
from typing import NamedTuple

from reddit_client import RedditPost


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SignalLevel(str, Enum):
    HIGH = "HIGH_SIGNAL"
    LOW = "LOW_SIGNAL"


class SignalType(str, Enum):
    ADOPTION = "adoption"           # people switching to / actively using X
    COMPLAINT = "complaint"         # frustration / pain points with X
    COMPARISON = "comparison"       # X vs Y discussions
    ANNOUNCEMENT = "announcement"   # new release / feature launch
    FEATURE_REQUEST = "feature_request"  # things people want from X
    PERFORMANCE = "performance"     # speed / memory / scaling issues
    PRICING = "pricing"             # cost / vendor frustration / lock-in
    GENERAL = "general"             # other substantive technical discussion


class ClassificationResult(NamedTuple):
    signal_level: SignalLevel
    signal_type: SignalType
    signal_score: float
    matched_tags: list[str]


# ---------------------------------------------------------------------------
# Tech vocabulary (also exported for use in analyze.py)
# ---------------------------------------------------------------------------

TECH_VOCABULARY: frozenset[str] = frozenset({
    # Languages
    "python", "rust", "go", "golang", "typescript", "javascript", "java",
    "kotlin", "swift", "dart", "zig", "ruby", "php", "scala", "haskell",
    "elixir", "erlang", "clojure", "cpp", "c++", "r",
    # Web frameworks
    "react", "vue", "svelte", "angular", "nextjs", "next.js", "nuxt", "remix",
    "astro", "solid", "htmx", "django", "fastapi", "flask", "rails", "laravel",
    "express", "fastify", "gin", "fiber", "actix", "axum", "spring",
    # Cloud & infra
    "aws", "gcp", "azure", "cloudflare", "vercel", "netlify", "fly.io",
    "railway", "render", "heroku", "digitalocean", "linode", "hetzner",
    "kubernetes", "k8s", "docker", "podman", "terraform", "pulumi", "ansible",
    "helm", "istio", "envoy", "nginx", "caddy", "traefik",
    # Databases
    "postgres", "postgresql", "mysql", "sqlite", "mongodb", "redis", "cassandra",
    "dynamodb", "supabase", "planetscale", "neon", "cockroachdb", "clickhouse",
    "elasticsearch", "opensearch", "pinecone", "weaviate", "chroma",
    # DevOps / CI-CD / observability
    "github actions", "jenkins", "gitlab ci", "circleci", "travis", "argo",
    "prometheus", "grafana", "datadog", "sentry", "opentelemetry", "jaeger",
    # AI / ML
    "langchain", "llamaindex", "llama", "openai", "anthropic", "mistral",
    "hugging face", "pytorch", "tensorflow", "jax", "onnx", "vllm",
    "ollama", "transformers", "langraph", "crewai", "autogen",
    # Messaging / streaming
    "kafka", "rabbitmq", "nats", "pulsar", "sqs", "pubsub", "eventbridge",
    # Protocols / standards / tools
    "graphql", "grpc", "rest", "websocket", "protobuf", "openapi", "swagger",
    "wasm", "webassembly", "ebpf", "nix", "bazel", "cmake",
    # High-value concepts
    "microservices", "monolith", "serverless", "edge computing",
    "rag", "fine-tuning", "embeddings", "llm", "llms", "api", "sdk",
    "vector db", "feature flag", "ci/cd",
})

# ---------------------------------------------------------------------------
# Signal pattern lists
# ---------------------------------------------------------------------------

_ADOPTION = [
    r"\bmigrat(?:ing|ed|ion)\b", r"\bswitch(?:ing|ed)?\s+(?:to|from|away)\b",
    r"\bmov(?:ing|ed)\s+(?:to|from|away)\b", r"\breplace[sd]?\b",
    r"\badopt(?:ing|ed|ion)\b", r"\bnow\s+using\b", r"\bwe\s+(?:use|switched|moved)\b",
    r"\bour\s+(?:team|stack|company)\s+(?:uses|switched|moved|went)\b",
    r"\bin\s+production\b", r"\bgoing\s+with\b", r"\bchose\b",
]

_COMPLAINT = [
    r"\bfrustrat(?:ing|ed|ion)\b", r"\bpain\s+point\b", r"\bannoy(?:ing|ed)\b",
    r"\bhate[sd]?\b", r"\bbroken\b", r"\bbuggy\b", r"\bunusable\b",
    r"\bawful\b", r"\bterrible\b", r"\bnightmare\b", r"\boverhead\b",
    r"\bbloat(?:ed)?\b", r"\babstraction\s+(?:hell|layer|overhead)\b",
    r"\bwhy\s+is\s+(?:this|it)\s+so\b", r"\bcan\'t\s+(?:believe|stand)\b",
    r"\bdealing\s+with\b", r"\bstruggling\s+with\b",
]

_COMPARISON = [
    r"\bvs\.?\b", r"\bversus\b", r"\bcompared?\s+to\b",
    r"\balternative\s+to\b", r"\bbetter\s+than\b", r"\bworse\s+than\b",
    r"\bsimilar\s+to\b", r"\binstead\s+of\b", r"\bprefer\b",
    r"\bpros?\s+and\s+cons?\b",
]

_ANNOUNCEMENT = [
    r"\breleas(?:ed?|ing)\b", r"\blaunch(?:ed|ing)?\b", r"\bannounce[sd]?\b",
    r"\bshipp(?:ed|ing)\b", r"\bintroduc(?:ing|ed)\b", r"\bv\d+\.\d+\b",
    r"\bopen[\s-]?sourc(?:ed?|ing)\b", r"\bnew\s+feature\b",
    r"\bbreaking\s+change\b", r"\bdeprecate[sd]?\b", r"\bga\b",
    r"\bgenerally\s+available\b", r"\bbeta\b",
]

_FEATURE_REQUEST = [
    r"\bwish(?:list)?\b", r"\bwould\s+(?:love|like)\b", r"\bfeature\s+request\b",
    r"\bplease\s+add\b", r"\bwhy\s+(?:doesn\'t|can\'t|won\'t|no)\b",
    r"\bmissing\s+(?:feature|support)\b", r"\bwant(?:ed)?\s+to\s+see\b",
]

_PERFORMANCE = [
    r"\bperformance\b", r"\blatency\b", r"\bthroughput\b", r"\bbenchmark\b",
    r"\bslow(?:er)?\b", r"\bfast(?:er)?\b", r"\bmemory\s+(?:leak|usage)\b",
    r"\bcpu\s+(?:usage|bound)\b", r"\bscaling?\b", r"\bbottleneck\b",
    r"\bload\s+(?:time|test|balancing)\b", r"\bp99\b", r"\bp95\b",
    r"\bms\s+latency\b", r"\boptimiz(?:ing|ation|ed)\b",
]

_PRICING = [
    r"\bpricin?g?\b", r"\bcost(?:ing|s)?\b", r"\bexpensive\b", r"\bcheap(?:er)?\b",
    r"\bvendor\s+lock[\s-]?in\b", r"\bbilling\b", r"\bsubscription\b",
    r"\bacquisition\b", r"\bfree\s+tier\b", r"\bpaid\s+plan\b",
    r"\bovercharge[sd]?\b",
]

_LOW_SIGNAL = [
    r"\bmeme\b", r"\bfunny\b", r"\bjoke\b", r"\bshitpost\b",
    r"\bhomework\b", r"\bassignment\b",
    r"\bpolitics\b", r"\bgovernment\b", r"\belection\b",
    r"\bnot\s+(?:tech|programming)\s+related\b",
]

_LOW_SIGNAL_TITLE = [
    r"^(?:eli5|explain\s+like|help\s+me\s+understand)\b",
    r"^rant:?\s",
    r"^(?:unpopular\s+opinion|hot\s+take):?\s",
]

_HYPE_WITHOUT_EVIDENCE = [
    r"\brevolution(?:ary)?\b", r"\bgame[- ]?changing\b",
    r"\bdisrupt(?:ive|ing)?\b", r"\bground[- ]?breaking\b",
]


def _compile_all(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_RE: dict[str, list[re.Pattern]] = {
    "adoption": _compile_all(_ADOPTION),
    "complaint": _compile_all(_COMPLAINT),
    "comparison": _compile_all(_COMPARISON),
    "announcement": _compile_all(_ANNOUNCEMENT),
    "feature_request": _compile_all(_FEATURE_REQUEST),
    "performance": _compile_all(_PERFORMANCE),
    "pricing": _compile_all(_PRICING),
    "low": _compile_all(_LOW_SIGNAL),
    "low_title": _compile_all(_LOW_SIGNAL_TITLE),
    "hype": _compile_all(_HYPE_WITHOUT_EVIDENCE),
}


def _hit_count(text: str, patterns: list[re.Pattern]) -> int:
    return sum(1 for p in patterns if p.search(text))


def _match_vocab(text: str) -> list[str]:
    tl = text.lower()
    return [term for term in TECH_VOCABULARY if term in tl]


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


def classify(post: RedditPost) -> ClassificationResult:
    """Return a ClassificationResult for a single Reddit post.

    Scoring approach:
    - Technical vocabulary hits anchor the signal (most important)
    - Signal-type patterns add bonus weight
    - Engagement metrics (score, comment count) contribute a small bonus
    - Low-signal and hype patterns subtract weight
    - Structural checks (very short posts, bare questions) can force LOW
    """
    full_text = f"{post.title} {post.selftext}"
    title_lower = post.title.lower()

    # Hard disqualifiers — check title-only low-signal markers
    low_title_hits = _hit_count(title_lower, _RE["low_title"])
    low_hits = _hit_count(full_text, _RE["low"])
    hype_hits = _hit_count(full_text, _RE["hype"])

    if low_hits >= 2:
        return ClassificationResult(SignalLevel.LOW, SignalType.GENERAL, 0.0, [])
    if low_title_hits >= 1 and post.score < 30 and not post.selftext.strip():
        return ClassificationResult(SignalLevel.LOW, SignalType.GENERAL, 0.0, [])

    # Tech vocabulary
    tech_tags = _match_vocab(full_text)
    tech_score = len(tech_tags) * 2.0

    # Signal-type pattern counts
    type_counts: dict[str, float] = {
        "adoption": _hit_count(full_text, _RE["adoption"]) * 3.0,
        "complaint": _hit_count(full_text, _RE["complaint"]) * 2.5,
        "comparison": _hit_count(full_text, _RE["comparison"]) * 3.0,
        "announcement": _hit_count(full_text, _RE["announcement"]) * 3.5,
        "feature_request": _hit_count(full_text, _RE["feature_request"]) * 2.0,
        "performance": _hit_count(full_text, _RE["performance"]) * 2.5,
        "pricing": _hit_count(full_text, _RE["pricing"]) * 2.0,
    }

    signal_score = (
        tech_score
        + sum(type_counts.values())
        - hype_hits * 1.5
        + min(post.score / 100.0, 5.0)
        + min(post.num_comments / 20.0, 3.0)
    )

    # Determine dominant signal type
    raw_counts = {
        SignalType.ADOPTION: type_counts["adoption"],
        SignalType.COMPLAINT: type_counts["complaint"],
        SignalType.COMPARISON: type_counts["comparison"],
        SignalType.ANNOUNCEMENT: type_counts["announcement"],
        SignalType.FEATURE_REQUEST: type_counts["feature_request"],
        SignalType.PERFORMANCE: type_counts["performance"],
        SignalType.PRICING: type_counts["pricing"],
    }
    best_type = max(raw_counts, key=raw_counts.get)
    if raw_counts[best_type] == 0.0:
        best_type = SignalType.GENERAL

    # Structural minimum: ignore sub-50 char posts with zero tech hits
    if len(full_text.strip()) < 50 and not tech_tags:
        return ClassificationResult(SignalLevel.LOW, SignalType.GENERAL, 0.0, [])

    level = SignalLevel.HIGH if signal_score >= 4.0 else SignalLevel.LOW
    return ClassificationResult(
        signal_level=level,
        signal_type=best_type,
        signal_score=round(signal_score, 2),
        matched_tags=tech_tags[:10],
    )
