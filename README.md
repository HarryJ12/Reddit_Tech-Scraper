# Reddit Tech Trend Monitor

Reddit has useful tech insights, but it's buried under noise. This tool filters the junk and surfaces real trends: what developers are actually adopting, complaining about, or switching away from.

Built in Python with three dependencies: `requests` (Reddit API), `pydantic` (data models), and `python-dotenv` (config). Everything else is stdlib.

---

## How It Works

1. **Fetch**: `reddit_client.py` pulls posts via Reddit's OAuth API OR public JSON endpoints. Rate-limited with a token bucket, responses are cached to disk in `.cache/` with a 5-hour TTL.

2. **Classify**: `tech_classifier.py` labels each post `HIGH_SIGNAL` or `LOW_SIGNAL` via keyword patterns. High-signal = specific tools, libraries, frameworks, infra. Low-signal = memes, politics, vague questions, hype.

3. **Collect**: `pipeline.py` orchestrates fetch + classify across all configured subreddits.

4. **Analyze**: `analyze.py` extracts named tech entities, clusters posts by entity, scores each cluster by engagement (`upvotes × 0.5 + comments × 0.3 + comment_upvotes × 0.2`), and generates a structured trend report.

5. **Output**: Ranked console summary. Writes Markdown and JSON to `output/`.

---

## Data Source

This tool can fetch Reddit data in two ways:

- **Without credentials** - Uses Reddit’s public `.json` endpoints (e.g. appending `.json` to subreddit URLs). No setup required, but rate limits are stricter.
- **With credentials** - Uses Reddit’s official API via OAuth for higher rate limits and more reliable access.

For lightweight usage or quick experiments, the `.json` approach is sufficient. For larger or repeated runs, credentials are recommended.

---

## Topic to Subreddit Mapping

`--topic` selects which subreddits to pull from; defined in `config.py`. There's no per-topic vocabulary. The same 150-term `TECH_VOCABULARY` and the same 7 signal-type patterns (adoption, complaint, comparison, etc.) run on every post regardless of topic. The domain scoping comes entirely from which subreddits you point it at, not from any segmented word list.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Credentials are optional, but they give slightly better rate limits. To get them: [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) -> create a **script** app -> add to `.env`:

```env
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=tech-trend-monitor/1.0 (tech research; your@email.com)
```

---

## Usage

```bash
python main.py --topic ai --top 5 --no-comments
python main.py --urls https://www.reddit.com/r/claude/.json
python main.py --subreddits programming, devops --days 7 --limit 50

```

**Topics:** `ai`, `ml`, `genai`, `programming`, `cloud`, `startups`, `tech`, `news`, `robotics`, `quantum`, `security`

| Flag | Default | Description |
|------|---------|-------------|
| `--topic` | `ai` group | Predefined subreddit group |
| `--subreddits` | `ai` group | Any comma-separated subreddit names |
| `--days` | 5 | How far back to look |
| `--limit` | 50 | Max posts per subreddit |
| `--top` | 10 | Trends to return |
| `--output FILE` | auto | `.md` or `.json` output |
| `--urls URL...` | — | Direct Reddit JSON listing URLs |
| `--no-comments` | off | Skip comment fetching |

---

## Project Structure

```
reddit-tech-trends/
├── main.py             # CLI entry point
├── config.py           # Subreddit groups, env vars, defaults
├── reddit_client.py    # API client, rate limiter, disk cache, data models
├── tech_classifier.py  # HIGH/LOW signal classification, signal type tagging
├── pipeline.py         # Orchestrates fetch + classify across subreddits
├── analyze.py          # Entity extraction, clustering, scoring, report generation
├── requirements.txt
├── .env.example
├── .cache/             # API response cache (auto-created)
└── output/             # Report output (auto-created)
```

---

## Using the Output with an external Chatbot

The `.md` report this tool generates can be fed into an LLM for a more summarized and conclusive analysis. Copy the contents of the report and use `SUMMARY_PROMPT.md` (included in this repo) as the system prompt any LLM of your choice (it is recommended to make a custom agent for repeatable use).

The prompt instructs the agent to:

- Merge overlapping trends and ignore low-signal noise
- Extract underlying problems from the raw evidence (performance, cost, reliability, usability)
- Surface hidden signals not explicitly stated in the report
- Generate specific opportunities based only on real problems
- Give a concrete action plan for anyone trying to keep up with AI

---

## Rate Limits

A typical run across 5 subreddits uses **200–350 requests**. Responses are cached for 5 hours so repeated runs within that window use **0 requests**. To reduce volume: `--no-comments` or lower `--limit`.
