from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Step:
    name: str
    args: List[str]
    optional: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full pipeline in order.")
    parser.add_argument(
        "--python", default=sys.executable, help="Python executable to use"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a failed step",
    )
    parser.add_argument(
        "--max-input-tokens", type=int, default=100000, help="LLM max input tokens"
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=40000, help="LLM max output tokens"
    )
    parser.add_argument(
        "--max-batches-per-request",
        type=int,
        default=100,
        help="LLM max batches per request",
    )
    return parser.parse_args()


def build_steps(args: argparse.Namespace) -> List[Step]:
    steps = [
        Step(
            "fetch",
            ["reddit/reddit_to_db.py", "--skip-existing", "--refresh-days", "7"],
        ),
        Step("embeddings", ["processing/embeddings.py", "--force"]),
        Step(
            "plot_embeddings",
            [
                "plots/plot_embeddings.py",
                "--plot-output",
                "plots/embeddings.png",
                "--save-db",
                "--umap-cluster-dim",
                "50",
            ],
        ),
        Step("cluster", ["modeling/cluster.py", "--clusterer", "hdbscan", "--save-db"]),
        Step("cluster_noise", ["modeling/cluster_noise.py"]),
        Step(
            "ctfidf",
            [
                "modeling/ctfidf_topics.py",
                "--output",
                "reports/ctfidf_topics.csv",
                "--top-n",
                "12",
                "--min-posts",
                "5",
                "--ngram-max",
                "2",
                "--min-df",
                "2",
                "--max-df",
                "0.95",
                "--stopwords-file",
                "config/stopwords_custom.txt",
            ],
        ),
        Step("topic_label_batch", ["llm/topic_label_batch.py", "--retry-missing", "3"]),
        Step(
            "export_missing",
            [
                "llm/export_missing_stance_sentiment_jsonl.py",
                "--output",
                "llm/stance_sentiment_missing.jsonl",
            ],
        ),
        Step(
            "stance_sentiment_batch",
            [
                "llm/stance_sentiment_batch.py",
                "--input",
                "llm/stance_sentiment_missing.jsonl",
                "--output",
                "llm/stance_sentiment_results2.jsonl",
                "--max-input-tokens",
                str(args.max_input_tokens),
                "--max-output-tokens",
                str(args.max_output_tokens),
                "--max-batches-per-request",
                str(args.max_batches_per_request),
            ],
        ),
        Step(
            "save_sentiment",
            [
                "llm/save_sentiment_stance_db.py",
                "--input",
                "llm/stance_sentiment_results2.jsonl",
                "--only-missing",
            ],
        ),
        Step("compute_weights", ["processing/weights.py", "--save-db"]),
    ]

    return steps


def format_cmd(cmd: List[str]) -> str:
    return subprocess.list2cmdline(cmd)


def run_steps(args: argparse.Namespace, steps: List[Step]) -> None:
    total = len(steps)
    for idx, step in enumerate(steps, start=1):
        cmd = [args.python] + step.args
        logger.info("Step %s/%s (%s): %s", idx, total, step.name, format_cmd(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            msg = f"Step {step.name} failed with exit code {result.returncode}."
            if step.optional or args.continue_on_error:
                logger.warning("%s Continuing.", msg)
                continue
            raise SystemExit(msg)


def main() -> None:
    args = parse_args()
    steps = build_steps(args)
    if not steps:
        logger.info("No steps to run.")
        return
    run_steps(args, steps)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
