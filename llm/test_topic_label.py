from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from llm.openrouter_client import OpenRouterClient  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_TEXT = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test OpenRouter topic labeling prompt."
    )
    parser.add_argument("--text", default=None, help="Input text for the prompt")
    parser.add_argument("--text-file", default=None, help="Read input text from file")
    parser.add_argument(
        "--prompt",
        default=str(REPO_ROOT / "llm" / "prompts" / "topic_label.j2"),
        help="Path to the topic label prompt template",
    )
    return parser.parse_args()


def render_prompt(template_path: Path, context: dict) -> str:
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


def load_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    return DEFAULT_TEXT


def main() -> None:
    args = parse_args()
    text = load_text(args)
    prompt_path = Path(args.prompt)

    prompt = render_prompt(prompt_path, {"text": text})

    client = OpenRouterClient()
    response = client.generate_text(prompt)

    print(response)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
