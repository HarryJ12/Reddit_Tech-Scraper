"""CLI entry point for the Reddit Tech Trend Monitor.

Usage examples:
  python main.py --subreddits programming,artificial,devops --days 7 --limit 50
  python main.py --subreddits startups,cloudcomputing --days 14 --limit 30 --output report.md
  python main.py --urls https://www.reddit.com/r/programming/new/.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from analyze import TrendAnalyzer
from config import settings
from pipeline import Pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify meaningful tech trends from Reddit discussions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--topic",
        choices=list(settings.topic_subreddits.keys()),
        metavar="TOPIC",
        help=(
            "Scrape a predefined topic group. Choices: "
            + ", ".join(settings.topic_subreddits.keys())
        ),
    )
    parser.add_argument(
        "--subreddits",
        default=None,
        help="Comma-separated subreddit names (default: config defaults)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        metavar="N",
        help="How many days back to look (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Max posts per subreddit (default: 50)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=settings.top_trends_count,
        metavar="N",
        help="Number of trends to surface (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write output to FILE (.json or .md). Defaults to output/ directory.",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        metavar="URL",
        help="One or more direct Reddit listing URLs (e.g. .../new/.json)",
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Skip comment fetching (faster, lower request volume)",
    )
    return parser.parse_args()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def main() -> int:
    args = _parse_args()
    _setup_logging()
    logger = logging.getLogger("main")

    if args.topic:
        subreddits = settings.topic_subreddits[args.topic]
    elif args.subreddits:
        subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    else:
        subreddits = settings.default_subreddits

    if not subreddits and not args.urls:
        logger.error("Provide --subreddits or --urls")
        return 1

    # ------------------------------------------------------------------ pipeline
    logger.info(
        "Starting pipeline | subreddits=%s | days=%d | limit=%d",
        subreddits or "(none)",
        args.days,
        args.limit,
    )
    pipeline = Pipeline(
        subreddits=subreddits,
        days=args.days,
        limit_per_subreddit=args.limit,
        fetch_comments=not args.no_comments,
        direct_urls=args.urls or [],
    )
    result = pipeline.run()

    if result.total_posts_fetched == 0:
        logger.error("No posts fetched — check credentials, subreddit names, or network.")
        return 1

    logger.info(
        "Collected %d posts (%d high-signal)",
        result.total_posts_fetched,
        len(result.high_signal_posts),
    )

    # ------------------------------------------------------------------ analysis
    analyzer = TrendAnalyzer(top_n=args.top)
    report = analyzer.analyze(result)

    if not report.trends:
        logger.warning(
            "No trends found. Try --days %d or more subreddits.", args.days * 2
        )
        print("\n[!] No trends identified with current settings.")
        print(
            f"    Fetched {result.total_posts_fetched} posts, "
            f"{len(result.high_signal_posts)} high-signal."
        )
        print("    Suggestions:")
        print("      • Increase --days (e.g. --days 14)")
        print("      • Increase --limit (e.g. --limit 100)")
        print("      • Add more subreddits (e.g. MachineLearning,devops)")
        return 0

    # ------------------------------------------------------------------ output
    if args.output:
        out_path = Path(args.output)
        suffix = out_path.suffix.lower()
        if suffix == ".json":
            out_path.write_text(analyzer.to_json(report))
        else:
            out_path.write_text(analyzer.to_markdown(report))
        print(f"Report saved → {out_path}")
    else:
        json_path, md_path = analyzer.save_reports(report)
        print(f"\nReports saved:")
        print(f"  JSON     → {json_path}")
        print(f"  Markdown → {md_path}")

    # Always print a brief summary to stdout
    print(f"\n{'='*60}")
    print(f"  Tech Trend Report — {report.generated_at.strftime('%Y-%m-%d')}")
    print(f"  Period: last {report.days} day(s)")
    print(f"  Subreddits: {', '.join(f'r/{s}' for s in report.subreddits[:5])}")
    print(f"  Posts analyzed: {report.total_posts_analyzed:,}  "
          f"(high-signal: {report.high_signal_posts:,})")
    print(f"{'='*60}\n")

    for i, trend in enumerate(report.trends, 1):
        confidence_icon = {"high": "●●●", "medium": "●●○", "low": "●○○"}.get(
            trend.confidence, "○○○"
        )
        print(f"  {i:2}. [{confidence_icon}] {trend.name}")
        print(f"       Signal: {trend.signal_type} | Posts: {trend.post_count} | "
              f"Score: {trend.weighted_score:.0f}")
        print(f"       {trend.what_is_happening[:120]}...")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
