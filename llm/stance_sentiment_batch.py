from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from llm.openrouter_client import OpenRouterClient  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress
    class _NullTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def update(self, n=1):
            return None

        def close(self):
            return None

        def set_postfix(self, **kwargs):
            return None

    def tqdm(*args, **kwargs):
        return _NullTqdm()

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = REPO_ROOT / "llm" / "prompts" / "stance_sentiment.txt"
DEFAULT_OUTPUT = REPO_ROOT / "llm" / "stance_sentiment_results.jsonl"
DEFAULT_SYSTEM = "You are a precise JSON classifier."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stance and sentiment classification via OpenRouter."
    )
    parser.add_argument("--input", help="JSONL/JSON input path with posts + comment lists")
    parser.add_argument("--post", help="Single post text (with --comment)")
    parser.add_argument("--comment", help="Single comment text (with --post)")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT), help="Prompt template path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL path")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "settings.yaml"),
        help="settings.yaml path",
    )
    parser.add_argument("--api-key", default=None, help="OpenRouter API key override")
    parser.add_argument("--model", default=None, help="OpenRouter model override")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature override")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Max output tokens override")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout override")
    parser.add_argument("--max-retries", type=int, default=None, help="Max retry count override")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--max-input-tokens", type=int, default=12000, help="Token budget per request")
    parser.add_argument(
        "--max-comments-per-batch",
        "--max-pairs-per-batch",
        dest="max_pairs_per_batch",
        type=int,
        default=4,
        help="Max comments per post batch (0 = no limit)",
    )
    parser.add_argument(
        "--max-batches-per-request",
        type=int,
        default=1,
        help="Max post batches per request (0 = no limit)",
    )
    parser.add_argument("--batch-delay-ms", type=int, default=0, help="Delay between requests in ms")
    parser.add_argument(
        "--batch-delay-every",
        type=int,
        default=0,
        help="Apply delay every N requests (0 = every request)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only show planned batches")
    parser.add_argument("--append", action="store_true", help="Append to output JSONL instead of overwrite")
    parser.add_argument("--debug-dir", default="", help="Optional dir for raw responses")
    parser.add_argument(
        "--re-run",
        action="store_true",
        help="Re-run failed batches using batch_*_input.json from --debug-dir",
    )
    return parser.parse_args()


def estimate_tokens(text: str) -> int:
    return max(1, int(math.ceil(len(text) / 4)))


def load_prompt_template(path: Path) -> str:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Prompt template is empty: {path}")
    if "{{pairs_json}}" not in raw:
        raise ValueError("Prompt template must include {{pairs_json}} placeholder.")
    return raw


def load_items(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if path.suffix.lower() == ".jsonl":
        items: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    return []


def normalize_comment(item: Dict[str, Any]) -> Dict[str, str]:
    return {
        "comment_id": str(item.get("comment_id") or ""),
        "comment_text": str(item.get("comment_text") or ""),
    }


def normalize_post(item: Dict[str, Any]) -> Dict[str, Any]:
    post_id = str(item.get("post_id") or "")
    post_text = str(item.get("post_text") or "")
    comments: List[Dict[str, str]] = []

    raw_comments = item.get("comments")
    if isinstance(raw_comments, list):
        for raw in raw_comments:
            if not isinstance(raw, dict):
                continue
            comment = normalize_comment(raw)
            if not comment["comment_id"] or not comment["comment_text"]:
                continue
            comments.append(comment)
    else:
        comment = normalize_comment(item)
        if comment["comment_id"] and comment["comment_text"]:
            comments.append(comment)

    return {
        "post_id": post_id,
        "post_text": post_text,
        "comments": comments,
    }


def group_posts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        post = normalize_post(raw)
        post_id = post["post_id"]
        if not post_id:
            continue
        if post_id not in grouped:
            grouped[post_id] = {
                "post_id": post_id,
                "post_text": post["post_text"],
                "comments": [],
            }
        group = grouped[post_id]
        if not group["post_text"] and post["post_text"]:
            group["post_text"] = post["post_text"]
        if post["comments"]:
            group["comments"].extend(post["comments"])
    return [post for post in grouped.values() if post["comments"]]


def build_prompt(template: str, posts: List[Dict[str, Any]]) -> str:
    payload = json.dumps(posts, ensure_ascii=False, indent=2)
    return template.replace("{{pairs_json}}", payload)


def split_post_batches(
    groups: List[Dict[str, Any]],
    template: str,
    max_input_tokens: int,
    max_pairs_per_batch: int,
) -> List[Dict[str, Any]]:
    batches: List[Dict[str, Any]] = []
    for group in groups:
        current: List[Dict[str, str]] = []
        for comment in group["comments"]:
            candidate = current + [comment]
            if max_pairs_per_batch and len(candidate) > max_pairs_per_batch:
                if current:
                    batches.append(
                        {
                            "post_id": group["post_id"],
                            "post_text": group["post_text"],
                            "comments": current,
                        }
                    )
                current = []
                candidate = [comment]

            if max_input_tokens > 0:
                prompt = build_prompt(
                    template,
                    [
                        {
                            "post_id": group["post_id"],
                            "post_text": group["post_text"],
                            "comments": candidate,
                        }
                    ],
                )
                tokens = estimate_tokens(prompt)
                if tokens > max_input_tokens:
                    if current:
                        batches.append(
                            {
                                "post_id": group["post_id"],
                                "post_text": group["post_text"],
                                "comments": current,
                            }
                        )
                        current = []
                        candidate = [comment]
                        prompt = build_prompt(
                            template,
                            [
                                {
                                    "post_id": group["post_id"],
                                    "post_text": group["post_text"],
                                    "comments": candidate,
                                }
                            ],
                        )
                        tokens = estimate_tokens(prompt)
                        if tokens > max_input_tokens:
                            logger.warning(
                                "Skipping comment %s for post %s; single comment exceeds token budget.",
                                comment["comment_id"],
                                group["post_id"],
                            )
                            continue
                    else:
                        logger.warning(
                            "Skipping comment %s for post %s; single comment exceeds token budget.",
                            comment["comment_id"],
                            group["post_id"],
                        )
                        continue

            current = candidate

        if current:
            batches.append(
                {
                    "post_id": group["post_id"],
                    "post_text": group["post_text"],
                    "comments": current,
                }
            )

    return batches


def build_request_batches(
    post_batches: List[Dict[str, Any]],
    template: str,
    max_input_tokens: int,
    max_batches_per_request: int,
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    current_post_batches: List[Dict[str, Any]] = []
    current_prompt = ""
    current_tokens = 0

    for batch in post_batches:
        candidate_batches = current_post_batches + [batch]

        if max_batches_per_request and len(candidate_batches) > max_batches_per_request:
            if current_post_batches:
                requests.append(
                    {
                        "post_batches": current_post_batches,
                        "prompt": current_prompt,
                        "tokens": current_tokens,
                    }
                )
            current_post_batches = []
            current_prompt = ""
            current_tokens = 0
            candidate_batches = [batch]

        prompt = build_prompt(template, candidate_batches)
        tokens = estimate_tokens(prompt)

        if max_input_tokens > 0 and tokens > max_input_tokens:
            if current_post_batches:
                requests.append(
                    {
                        "post_batches": current_post_batches,
                        "prompt": current_prompt,
                        "tokens": current_tokens,
                    }
                )
                current_post_batches = []
                current_prompt = ""
                current_tokens = 0

                candidate_batches = [batch]
                prompt = build_prompt(template, candidate_batches)
                tokens = estimate_tokens(prompt)
                if max_input_tokens > 0 and tokens > max_input_tokens:
                    logger.warning(
                        "Skipping batch for post %s; exceeds token budget.",
                        batch["post_id"],
                    )
                    continue
            else:
                logger.warning(
                    "Skipping batch for post %s; exceeds token budget.",
                    batch["post_id"],
                )
                continue

        current_post_batches = candidate_batches
        current_prompt = prompt
        current_tokens = tokens

    if current_post_batches:
        requests.append(
            {
                "post_batches": current_post_batches,
                "prompt": current_prompt,
                "tokens": current_tokens,
            }
        )

    return requests


def parse_response(raw: str) -> List[Dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            return data["items"]
        if isinstance(data.get("results"), list):
            return data["results"]
        return [data]
    return []


def write_jsonl(path: Path, rows: List[Dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_sleep(batch_idx: int, batch_delay_ms: int, batch_delay_every: int) -> None:
    if batch_delay_ms <= 0:
        return
    if batch_delay_every > 0:
        if batch_idx % batch_delay_every == 0:
            time.sleep(batch_delay_ms / 1000)
    else:
        time.sleep(batch_delay_ms / 1000)


def count_comments(post_batches: List[Dict[str, Any]]) -> int:
    return sum(len(batch.get("comments") or []) for batch in post_batches)


def _debug_input_sort_key(path: Path) -> Tuple[int, str]:
    parts = path.stem.split("_")
    for part in parts:
        if part.isdigit():
            return int(part), path.name
    return 0, path.name


def merge_post_batches(batches: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    seen_comments: Dict[str, set[str]] = {}
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        post_id = str(batch.get("post_id") or "")
        if not post_id:
            continue
        post_text = str(batch.get("post_text") or "")
        if post_id not in grouped:
            grouped[post_id] = {
                "post_id": post_id,
                "post_text": post_text,
                "comments": [],
            }
            seen_comments[post_id] = set()
        group = grouped[post_id]
        if not group["post_text"] and post_text:
            group["post_text"] = post_text
        for comment in batch.get("comments") or []:
            if not isinstance(comment, dict):
                continue
            comment_id = str(comment.get("comment_id") or "")
            comment_text = str(comment.get("comment_text") or "")
            if not comment_id or not comment_text:
                continue
            if comment_id in seen_comments[post_id]:
                continue
            seen_comments[post_id].add(comment_id)
            group["comments"].append(
                {
                    "comment_id": comment_id,
                    "comment_text": comment_text,
                }
            )
    return list(grouped.values())


def load_debug_posts(debug_dir: Path) -> List[Dict[str, Any]]:
    if not debug_dir.exists():
        raise FileNotFoundError(f"Debug dir not found: {debug_dir}")
    input_files = sorted(
        debug_dir.glob("batch_*_input.json"),
        key=_debug_input_sort_key,
    )
    if not input_files:
        logger.info("No debug input files found in %s", debug_dir)
        return []

    batches: List[Dict[str, Any]] = []
    for path in input_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping invalid JSON in %s", path)
            continue
        if not isinstance(data, list):
            logger.warning("Skipping non-list debug input in %s", path)
            continue
        for item in data:
            if isinstance(item, dict):
                batches.append(item)

    logger.info("Loaded %s debug batches from %s files.", len(batches), len(input_files))
    return merge_post_batches(batches)


def build_debug_meta(
    batch: Dict[str, Any],
    *,
    expected: int,
    received: int,
    reason: str,
    request_index: int,
) -> Dict[str, Any]:
    posts_meta = []
    for post in batch.get("post_batches") or []:
        comment_ids = [
            str(item.get("comment_id") or "")
            for item in (post.get("comments") or [])
            if item.get("comment_id")
        ]
        posts_meta.append(
            {
                "post_id": str(post.get("post_id") or ""),
                "comment_ids": comment_ids,
            }
        )
    return {
        "request_index": request_index,
        "reason": reason,
        "expected_comments": expected,
        "received_comments": received,
        "post_batch_count": len(batch.get("post_batches") or []),
        "posts": posts_meta,
    }


def main() -> None:
    args = parse_args()

    if args.re_run:
        if not args.debug_dir:
            raise SystemExit("--re-run requires --debug-dir")
        template = load_prompt_template(Path(args.prompt))
        groups = load_debug_posts(Path(args.debug_dir))
        if not groups:
            logger.info("No debug posts found to re-run.")
            return
    else:
        if not args.input and not (args.post and args.comment):
            raise SystemExit("Provide --input or --post + --comment")

        if args.input:
            items = load_items(Path(args.input))
        else:
            items = [
                {
                    "post_id": "post_1",
                    "post_text": args.post,
                    "comments": [
                        {
                            "comment_id": "comment_1",
                            "comment_text": args.comment,
                        }
                    ],
                }
            ]

        if not items:
            logger.info("No items found to process.")
            return

        template = load_prompt_template(Path(args.prompt))
        groups = group_posts(items)
        if not groups:
            logger.info("No posts with comments found to process.")
            return

    max_input_tokens = int(args.max_input_tokens or 0)
    max_comments_per_batch = int(args.max_pairs_per_batch or 0)
    max_batches_per_request = int(args.max_batches_per_request or 0)

    post_batches = split_post_batches(
        groups,
        template,
        max_input_tokens,
        max_comments_per_batch,
    )
    requests = build_request_batches(
        post_batches,
        template,
        max_input_tokens,
        max_batches_per_request,
    )

    total_comments = sum(len(group.get("comments") or []) for group in groups)
    logger.info(
        "Prepared %s requests from %s posts (%s comments, %s post batches).",
        len(requests),
        len(groups),
        total_comments,
        len(post_batches),
    )

    if args.dry_run:
        for idx, batch in enumerate(requests, start=1):
            comment_count = count_comments(batch["post_batches"])
            logger.info(
                "Request %s: comments=%s batches=%s tokens=%s",
                idx,
                comment_count,
                len(batch["post_batches"]),
                batch["tokens"],
            )
        return

    client = OpenRouterClient(
        config_path=Path(args.config),
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)
    total_requests = len(requests)
    total_comments_in_requests = sum(
        count_comments(request["post_batches"]) for request in requests
    )
    processed_comments = 0
    progress = tqdm(total=total_requests, desc="OpenRouter requests", unit="req")
    debug_prefix = "rerun_" if args.re_run else ""
    try:
        for batch_idx, batch in enumerate(requests, start=1):
            response = client.generate_text(
                batch["prompt"],
                system=args.system,
                max_tokens=args.max_output_tokens,
            )

            if debug_dir:
                debug_path = debug_dir / f"{debug_prefix}batch_{batch_idx}.txt"
                debug_path.write_text(response or "", encoding="utf-8")

            parsed = parse_response(response)
            expected = count_comments(batch["post_batches"])
            received = len(parsed)
            error_reason = ""
            if not parsed:
                error_reason = "empty_or_invalid_json"
                logger.warning("Empty/invalid response for request %s", batch_idx)
            elif received != expected:
                error_reason = "count_mismatch"
                logger.warning(
                    "Request %s: expected %s items, got %s",
                    batch_idx,
                    expected,
                    received,
                )

            if error_reason and debug_dir:
                meta = build_debug_meta(
                    batch,
                    expected=expected,
                    received=received,
                    reason=error_reason,
                    request_index=batch_idx,
                )
                meta_path = debug_dir / f"{debug_prefix}batch_{batch_idx}_meta.json"
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                input_path = debug_dir / f"{debug_prefix}batch_{batch_idx}_input.json"
                input_path.write_text(
                    json.dumps(batch["post_batches"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            if parsed:
                write_jsonl(output_path, parsed, append=args.append or batch_idx > 1)

            processed_comments += expected
            remaining_requests = total_requests - batch_idx
            remaining_comments = total_comments_in_requests - processed_comments
            progress.set_postfix(
                remaining_requests=remaining_requests,
                remaining_comments=remaining_comments,
            )
            progress.update(1)
            maybe_sleep(batch_idx, int(args.batch_delay_ms or 0), int(args.batch_delay_every or 0))
    finally:
        progress.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
