from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from llm.openrouter_client import OpenRouterClient  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch topic labeling from chunk/post CSVs with multi-chunk LLM calls."
    )
    parser.add_argument(
        "--chunks-csv",
        default="reports/topic_label_chunks.csv",
        help="Chunks CSV path",
    )
    parser.add_argument(
        "--posts-csv",
        default="reports/topic_label_posts.csv",
        help="Posts CSV path",
    )
    parser.add_argument(
        "--prompt",
        default=str(REPO_ROOT / "llm" / "prompts" / "topic_label_batch.j2"),
        help="Prompt template path",
    )
    parser.add_argument("--max-input-tokens", type=int, default=20000, help="Token budget per call")
    parser.add_argument("--max-chunks-per-call", type=int, default=3, help="Max chunks per LLM call")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Max output tokens override")
    parser.add_argument("--topic-id", default=None, help="Only process a single topic_id")
    parser.add_argument(
        "--topic-ids",
        nargs="+",
        default=None,
        help="Only process the specified topic_ids (space-separated)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of chunks")
    parser.add_argument("--dry-run", action="store_true", help="Only show planned batches")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append results to the output CSV instead of overwriting",
    )
    parser.add_argument(
        "--output",
        default="reports/topic_label_results.csv",
        help="Output CSV for LLM results",
    )
    parser.add_argument(
        "--debug-dir",
        default="",
        help="Optional directory to write raw LLM responses",
    )
    return parser.parse_args()


def render_prompt(template_path: Path, chunks_text: str) -> str:
    try:
        from jinja2 import Template
    except ImportError as exc:
        raise SystemExit("jinja2 is required to render prompts. Install it first.") from exc

    raw = template_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Prompt template is empty: {template_path}")

    return Template(raw).render(chunks=chunks_text).strip()


def estimate_tokens(text: str) -> int:
    return max(1, int(math.ceil(len(text) / 4)))


def load_chunks(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    chunks: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            topic_id = (row.get("topic_id") or "").strip()
            chunk_id_raw = (row.get("chunk_id") or "").strip()
            if not topic_id or not chunk_id_raw:
                continue
            try:
                chunk_id = int(chunk_id_raw)
            except ValueError:
                continue
            ctfidf_terms = (row.get("ctfidf_terms") or "").strip()
            chunks[(topic_id, chunk_id)] = {
                "topic_id": topic_id,
                "chunk_id": chunk_id,
                "ctfidf_terms": ctfidf_terms,
            }
    return chunks


def load_posts(path: Path) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            topic_id = (row.get("topic_id") or "").strip()
            chunk_id_raw = (row.get("chunk_id") or "").strip()
            if not topic_id or not chunk_id_raw:
                continue
            try:
                chunk_id = int(chunk_id_raw)
            except ValueError:
                continue
            post_index_raw = (row.get("post_index") or "").strip()
            try:
                post_index = int(post_index_raw)
            except ValueError:
                post_index = 0
            grouped.setdefault((topic_id, chunk_id), []).append(
                {
                    "post_index": post_index,
                    "tag": (row.get("tag") or "").strip(),
                    "post_id": (row.get("post_id") or "").strip(),
                    "distance": (row.get("distance") or "").strip(),
                    "text": (row.get("text") or "").strip(),
                }
            )
    for key, entries in grouped.items():
        entries.sort(key=lambda item: item["post_index"])
    return grouped


def build_entry(entry: Dict[str, Any]) -> str:
    index = entry["post_index"]
    tag = entry["tag"]
    post_id = entry["post_id"]
    distance = entry["distance"]
    text = entry["text"]
    return f"[{index}] tag={tag} post_id={post_id} distance={distance}\n{text}"


def build_chunk_block(
    *,
    topic_id: str,
    chunk_id: int,
    ctfidf_terms: str,
    entries: List[Dict[str, Any]],
) -> str:
    lines = [f"CHUNK chunk_id={chunk_id} topic_id={topic_id}"]
    if ctfidf_terms:
        lines.append(f"CTFIDF_TERMS: {ctfidf_terms}")
    lines.append("POSTS (tagged NEAR/FAR; lower distance = more central):")
    post_lines = [build_entry(entry) for entry in entries]
    lines.append("\n\n".join(post_lines))
    return "\n".join(lines).strip()


def iter_chunk_items(
    chunks_meta: Dict[Tuple[str, int], Dict[str, Any]],
    posts_by_chunk: Dict[Tuple[str, int], List[Dict[str, Any]]],
    *,
    topic_id_filter: Optional[str],
    topic_ids_filter: Optional[List[str]],
    limit: int,
) -> Iterable[Dict[str, Any]]:
    keys = sorted(chunks_meta.keys(), key=lambda item: (item[0], item[1]))
    count = 0
    for key in keys:
        if limit and count >= limit:
            break
        meta = chunks_meta.get(key)
        if not meta:
            continue
        topic_id, chunk_id = key
        if topic_id_filter and topic_id != topic_id_filter:
            continue
        if topic_ids_filter and topic_id not in topic_ids_filter:
            continue
        entries = posts_by_chunk.get(key)
        if not entries:
            continue
        block = build_chunk_block(
            topic_id=topic_id,
            chunk_id=chunk_id,
            ctfidf_terms=meta.get("ctfidf_terms", ""),
            entries=entries,
        )
        count += 1
        yield {
            "topic_id": topic_id,
            "chunk_id": chunk_id,
            "block": block,
        }


def build_batches(
    items: List[Dict[str, Any]],
    *,
    prompt_path: Path,
    max_input_tokens: int,
    max_chunks_per_call: int,
) -> List[Dict[str, Any]]:
    batches: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    current_blocks: List[str] = []
    current_tokens = 0

    for item in items:
        candidate_blocks = current_blocks + [item["block"]]
        if max_chunks_per_call and len(candidate_blocks) > max_chunks_per_call:
            if current:
                batches.append(
                    {
                        "chunks": current,
                        "prompt": render_prompt(prompt_path, "\n\n---\n\n".join(current_blocks)),
                        "tokens": current_tokens,
                    }
                )
            current = []
            current_blocks = []
            current_tokens = 0
            candidate_blocks = [item["block"]]

        prompt = render_prompt(prompt_path, "\n\n---\n\n".join(candidate_blocks))
        tokens = estimate_tokens(prompt)

        if tokens > max_input_tokens:
            if current:
                batches.append(
                    {
                        "chunks": current,
                        "prompt": render_prompt(prompt_path, "\n\n---\n\n".join(current_blocks)),
                        "tokens": current_tokens,
                    }
                )
                current = []
                current_blocks = []
                current_tokens = 0

                prompt = render_prompt(prompt_path, item["block"])
                tokens = estimate_tokens(prompt)
                if tokens > max_input_tokens:
                    logger.warning(
                        "Skipping chunk %s/%s; single chunk exceeds token budget.",
                        item["topic_id"],
                        item["chunk_id"],
                    )
                    continue
                current = [item]
                current_blocks = [item["block"]]
                current_tokens = tokens
            else:
                logger.warning(
                    "Skipping chunk %s/%s; single chunk exceeds token budget.",
                    item["topic_id"],
                    item["chunk_id"],
                )
            continue

        current = current + [item]
        current_blocks = candidate_blocks
        current_tokens = tokens

    if current:
        batches.append(
            {
                "chunks": current,
                "prompt": render_prompt(prompt_path, "\n\n---\n\n".join(current_blocks)),
                "tokens": current_tokens,
            }
        )

    return batches


def parse_response(raw: str) -> List[Dict[str, Any]]:
    text = raw.strip()
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

    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return data["results"]
        return [data]
    if isinstance(data, list):
        return data
    return []


def write_results(path: Path, rows: List[Dict[str, Any]], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    write_header = not append or not path.exists() or path.stat().st_size == 0
    with path.open(mode, encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "batch_id",
                "topic_id",
                "chunk_id",
                "topic_name",
                "topic_description",
                "confidence",
                "representative_indices",
                "status",
                "error",
            ],
        )
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    chunks_path = Path(args.chunks_csv)
    posts_path = Path(args.posts_csv)
    prompt_path = Path(args.prompt)

    chunks_meta = load_chunks(chunks_path)
    posts_by_chunk = load_posts(posts_path)
    items = list(
        iter_chunk_items(
            chunks_meta,
            posts_by_chunk,
            topic_id_filter=args.topic_id,
            topic_ids_filter=args.topic_ids,
            limit=int(args.limit or 0),
        )
    )
    if not items:
        logger.info("No chunk items found to label.")
        return

    batches = build_batches(
        items,
        prompt_path=prompt_path,
        max_input_tokens=args.max_input_tokens,
        max_chunks_per_call=args.max_chunks_per_call,
    )
    logger.info("Prepared %s batches from %s chunks.", len(batches), len(items))

    if args.dry_run:
        for idx, batch in enumerate(batches, start=1):
            logger.info(
                "Batch %s: chunks=%s tokens=%s",
                idx,
                len(batch["chunks"]),
                batch["tokens"],
            )
        return

    client = OpenRouterClient()
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for batch_idx, batch in enumerate(batches, start=1):
        prompt = batch["prompt"]
        response = client.generate_text(
            prompt,
            system="You are an expert topic labeler.",
            max_tokens=args.max_output_tokens,
        )
        if debug_dir:
            debug_path = debug_dir / f"batch_{batch_idx}.txt"
            debug_path.write_text(response or "", encoding="utf-8")

        parsed = parse_response(response)
        parsed_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for item in parsed:
            topic_id = str(item.get("topic_id") or "").strip()
            chunk_id_raw = str(item.get("chunk_id") or "").strip()
            if not topic_id or not chunk_id_raw:
                continue
            try:
                chunk_id = int(chunk_id_raw)
            except ValueError:
                continue
            parsed_map[(topic_id, chunk_id)] = item

        for chunk in batch["chunks"]:
            key = (chunk["topic_id"], int(chunk["chunk_id"]))
            item = parsed_map.get(key)
            if not item:
                results.append(
                    {
                        "batch_id": batch_idx,
                        "topic_id": key[0],
                        "chunk_id": key[1],
                        "topic_name": "",
                        "topic_description": "",
                        "confidence": "",
                        "representative_indices": "",
                        "status": "error",
                        "error": "missing_result",
                    }
                )
                continue

            rep_indices = item.get("representative_indices")
            results.append(
                {
                    "batch_id": batch_idx,
                    "topic_id": key[0],
                    "chunk_id": key[1],
                    "topic_name": str(item.get("topic_name") or ""),
                    "topic_description": str(item.get("topic_description") or ""),
                    "confidence": str(item.get("confidence") or ""),
                    "representative_indices": json.dumps(rep_indices)
                    if isinstance(rep_indices, list)
                    else str(rep_indices or ""),
                    "status": "ok",
                    "error": "",
                }
            )

    output_path = Path(args.output)
    append = args.append or bool(args.topic_id) or bool(args.topic_ids)
    write_results(output_path, results, append=append)
    logger.info("Wrote %s results to %s", len(results), output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
