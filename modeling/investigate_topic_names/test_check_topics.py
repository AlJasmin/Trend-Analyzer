from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from db.store import CONFIG_PATH, connect_from_config  # noqa: E402
from llm.openrouter_client import OpenRouterClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual test for check_topics.j2 prompt."
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="Path to settings.yaml"
    )
    parser.add_argument(
        "--prompt",
        default=str(REPO_ROOT / "llm" / "prompts" / "check_topics.j2"),
        help="Prompt template path",
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Use compact prompt (post_id, matches, fit_score only)",
    )
    parser.add_argument(
        "--post-id", default=None, help="Load topic fields from DB by post_id"
    )
    parser.add_argument("--topic-name", default=None, help="Topic name override")
    parser.add_argument(
        "--topic-description", default=None, help="Topic description override"
    )
    parser.add_argument("--topic-text", default=None, help="Topic text override")
    parser.add_argument("--text-file", default=None, help="Read topic_text from file")
    parser.add_argument(
        "--print-prompt", action="store_true", help="Print rendered prompt"
    )
    return parser.parse_args()


def render_prompt(template_path: Path, context: Dict[str, str]) -> str:
    try:
        from jinja2 import Template
    except ImportError as exc:
        raise SystemExit(
            "jinja2 is required to render prompts. Install it first."
        ) from exc

    raw = template_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Prompt template is empty: {template_path}")

    return Template(raw).render(**context).strip()


def build_entry(
    post_id: str, topic_name: str, topic_description: str, topic_text: str
) -> Dict[str, str]:
    return {
        "post_id": str(post_id),
        "topic_name": str(topic_name),
        "topic_description": str(topic_description),
        "topic_text": str(topic_text),
    }


def dump_entries(entries: list[Dict[str, Any]]) -> str:
    return json.dumps(entries, ensure_ascii=True, separators=(",", ":"))


def parse_json(raw: str) -> Optional[Any]:
    cleaned = raw.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            segment = part.strip()
            if segment.startswith("json"):
                segment = segment[4:].lstrip()
            if segment.startswith("{") or segment.startswith("["):
                cleaned = segment
                break
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def main() -> None:
    args = parse_args()
    topic_name = args.topic_name or ""
    topic_description = args.topic_description or ""
    topic_text = args.topic_text or ""
    post_id = "manual"

    if args.text_file:
        topic_text = Path(args.text_file).read_text(encoding="utf-8")

    if args.post_id:
        store = connect_from_config(Path(args.config))
        try:
            doc = store.posts.find_one({"post_id": args.post_id})
        finally:
            store.close()
        if not doc:
            raise SystemExit(f"post_id not found: {args.post_id}")
        post_id = str(doc.get("post_id") or args.post_id)
        topic_name = args.topic_name or str(doc.get("topic_name") or "")
        topic_description = args.topic_description or str(
            doc.get("topic_description") or ""
        )
        topic_text = args.topic_text or str(doc.get("topic_text") or "")

    if not topic_text:
        raise SystemExit(
            "topic_text is required (use --topic-text, --text-file, or --post-id)"
        )

    entry = build_entry(post_id, topic_name, topic_description, topic_text)
    payload = dump_entries([entry])
    prompt_path = Path(args.prompt)
    if args.compact_output:
        prompt_path = REPO_ROOT / "llm" / "prompts" / "check_topics_compact.j2"
    prompt = render_prompt(prompt_path, {"POSTS_JSON": payload})
    if args.print_prompt:
        print(prompt)
        print("\n---\n")

    client = OpenRouterClient(config_path=Path(args.config))
    response = client.generate_text(prompt)
    print(response)

    parsed = parse_json(response)
    if parsed is not None:
        print("\nParsed JSON:\n")
        print(json.dumps(parsed, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
