import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from db.store import CONFIG_PATH, connect_from_config  # noqa: E402
from reddit.reddit_cleaner import build_topic_text  # noqa: E402

logger = logging.getLogger(__name__)

# A strong, common NLI model. Works well for English.
# Alternatives you can try:
# - "roberta-large-mnli"
# - "microsoft/deberta-v3-large-mnli" (often very good, heavier)
MODEL_NAME = "facebook/bart-large-mnli"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

id2label = model.config.id2label  # e.g. {0:'contradiction',1:'neutral',2:'entailment'}
label2id = model.config.label2id


def nli_probs(premise: str, hypothesis: str) -> dict:
    """
    Returns probabilities for contradiction/neutral/entailment for a single (premise, hypothesis).
    """
    inputs = tokenizer(
        premise, hypothesis, return_tensors="pt", truncation=True, max_length=256
    )
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = F.softmax(logits, dim=-1)

    out = {id2label[i].lower(): float(probs[i].item()) for i in range(probs.shape[0])}
    # Ensure keys exist even if model uses different casing
    return {
        "contradiction": out.get("contradiction", 0.0),
        "neutral": out.get("neutral", 0.0),
        "entailment": out.get("entailment", 0.0),
    }


def stance_to_post(post: str, comment: str, threshold: float = 0.55) -> dict:
    """
    Predict stance of 'comment' towards 'post'.

    Strategy:
    - Premise: the comment (what the user said)
    - Hypotheses: "I support the post." / "I oppose the post."
    - Map NLI outputs to stance:
        support_hyp: entailment -> FAVOR, contradiction -> AGAINST
        oppose_hyp: entailment -> AGAINST, contradiction -> FAVOR
      Neutral if neither is confident enough.

    threshold: confidence threshold for deciding FAVOR/AGAINST. Otherwise NONE.
    """
    premise = f"POST: {post}\nCOMMENT: {comment}"

    # Two hypotheses about the commenter's stance towards the post
    hyp_support = "The commenter supports the post."
    hyp_oppose = "The commenter opposes the post."

    p_support = nli_probs(premise, hyp_support)
    p_oppose = nli_probs(premise, hyp_oppose)

    # Convert to stance scores (simple & robust mapping)
    favor_score = max(p_support["entailment"], p_oppose["contradiction"])
    against_score = max(p_oppose["entailment"], p_support["contradiction"])
    none_score = max(p_support["neutral"], p_oppose["neutral"])

    # Decide label
    best = max(
        ("FAVOR", favor_score),
        ("AGAINST", against_score),
        ("NONE", none_score),
        key=lambda x: x[1],
    )

    # Optional: require confidence for FAVOR/AGAINST, else NONE
    if best[0] in ("FAVOR", "AGAINST") and best[1] < threshold:
        label = "NONE"
        score = none_score
    else:
        label, score = best[0], best[1]

    return {
        "label": label,
        "score": float(score),
        "scores": {
            "FAVOR": float(favor_score),
            "AGAINST": float(against_score),
            "NONE": float(none_score),
        },
        "debug": {
            "support_hyp": p_support,
            "oppose_hyp": p_oppose,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stance test on real DB samples.")
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="Path to settings.yaml"
    )
    parser.add_argument(
        "--sample", type=int, default=40, help="Number of post/comment pairs"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55, help="Confidence threshold"
    )
    parser.add_argument(
        "--max-chars", type=int, default=400, help="Max chars to print per text"
    )
    return parser.parse_args()


def shorten(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def fetch_pairs(sample_size: int, config_path: Path) -> List[Dict[str, str]]:
    store = connect_from_config(config_path)
    try:
        pipeline = [
            {
                "$match": {
                    "post_id": {"$exists": True, "$nin": [None, ""]},
                    "$or": [
                        {"comment_text_clean": {"$exists": True, "$ne": ""}},
                        {"comment_text": {"$exists": True, "$ne": ""}},
                        {"body_clean": {"$exists": True, "$ne": ""}},
                        {"body": {"$exists": True, "$ne": ""}},
                    ],
                }
            },
            {
                "$lookup": {
                    "from": "posts",
                    "localField": "post_id",
                    "foreignField": "post_id",
                    "as": "post",
                }
            },
            {"$unwind": "$post"},
            {
                "$match": {
                    "$or": [
                        {"post.topic_text": {"$exists": True, "$ne": ""}},
                        {"post.title": {"$exists": True, "$ne": ""}},
                        {"post.selftext": {"$exists": True, "$ne": ""}},
                    ]
                }
            },
            {"$sample": {"size": int(sample_size)}},
            {
                "$project": {
                    "_id": 0,
                    "comment_id": 1,
                    "post_id": 1,
                    "comment_text_clean": 1,
                    "comment_text": 1,
                    "body_clean": 1,
                    "body": 1,
                    "post": {
                        "topic_text": 1,
                        "title": 1,
                        "selftext": 1,
                    },
                }
            },
        ]
        docs = list(store.comments.aggregate(pipeline))
    finally:
        store.close()

    pairs: List[Dict[str, str]] = []
    for doc in docs:
        comment = (
            doc.get("comment_text_clean")
            or doc.get("comment_text")
            or doc.get("body_clean")
            or doc.get("body")
            or ""
        ).strip()
        post_doc = doc.get("post") or {}
        post_text = (post_doc.get("topic_text") or "").strip()
        if not post_text:
            title = post_doc.get("title") or ""
            selftext = post_doc.get("selftext") or ""
            post_text = build_topic_text(title, selftext, bool(selftext))
        if not comment or not post_text:
            continue
        pairs.append(
            {
                "post_id": str(doc.get("post_id") or ""),
                "comment_id": str(doc.get("comment_id") or ""),
                "post_text": post_text,
                "comment_text": comment,
            }
        )
    return pairs


def main() -> None:
    args = parse_args()
    pairs = fetch_pairs(args.sample, Path(args.config))
    if not pairs:
        logger.info("No post/comment pairs found.")
        return

    for idx, pair in enumerate(pairs, start=1):
        res = stance_to_post(
            pair["post_text"], pair["comment_text"], threshold=args.threshold
        )
        print("-" * 80)
        print(f"{idx:02d} POST_ID   :", pair["post_id"])
        print(f"{idx:02d} COMMENT_ID:", pair["comment_id"])
        print("POST   :", shorten(pair["post_text"], args.max_chars))
        print("COMMENT:", shorten(pair["comment_text"], args.max_chars))
        print("PRED   :", res["label"], f"(score={res['score']:.3f})")
        print("SCORES :", {k: round(v, 3) for k, v in res["scores"].items()})


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
