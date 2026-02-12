# Trend-Analyzer (Reddit Discourse & Topic Pipeline)

## Motivation

Scientific and tech topics are discussed not only in papers and news, but also on platforms like Reddit. These discussions help to understand:

- what people talk about
- how they react (sentiment)
- where debates split into pro vs. contra (polarization)

---

## Overview

An end-to-end pipeline that turns Reddit discussions into structured topics and metrics. It produces:

- topic clusters (`topic_id`)
- topic keywords (c-TF-IDF)
- LLM topic labels (name, description, confidence)
- comment sentiment + stance (batched LLM calls)
- weighted post-level distributions (community signal + model confidence)
- polarization score (when enough stance-labeled comments exist)

Data is stored in MongoDB and visualized in a Next.js dashboard (live DB or snapshot).

---

## What the pipeline does

1. Collect and prepare data. Fetch Reddit posts and comments from selected subreddits, clean the text, remove links and image-only posts, and build a topic text from title + cleaned body.
2. Embeddings. Convert each post into a vector representation (SentenceTransformers).
3. Dimensionality reduction (UMAP). Project embeddings for clustering and plotting.
4. Clustering (HDBSCAN or DBSCAN). Assign a `topic_id` based on similarity. The main pipeline defaults to HDBSCAN.
5. Topic keywords (c-TF-IDF). Compute terms that are frequent inside a topic and rare across others. Export to CSV.
6. Topic labeling (LLM). Use c-TF-IDF terms and representative posts from each cluster to prompt the LLM. Save name, description, and confidence.
7. Sentiment + stance (LLM on comments). Process only comments without labels, batch requests to stay within token limits, and store results in MongoDB.
8. Weighting and aggregation. Compute comment weights from upvotes and LLM confidence, then aggregate weighted sentiment and stance per post. Compute polarization when enough stance-labeled comments exist (default minimum: 5).
9. Incremental refresh. Re-runs can skip existing items and focus on missing or recently updated data.

---

## Key metrics

**Weighted Sentiment / Stance**
Not every comment counts equally. A comment weighs more if it has more upvotes and/or the model is more confident.

**Polarization Score**
A post/topic is more polarized when both agree and disagree are strong at the same time. It is best used for relative comparison across topics.

---

## Features

- Configurable Reddit scraping (filters, categories, refresh window)
- MongoDB storage with indexes
- Embeddings + long-text handling (chunking)
- UMAP projection for clustering and plots
- HDBSCAN or DBSCAN topic clustering
- c-TF-IDF topic keywords (CSV export)
- LLM topic labeling (name/description/confidence)
- LLM stance + sentiment on comments (batched, retry missing)
- Weighting via upvotes + confidence
- Polarization score (when enough stance-labeled data exists)
- Dashboard (live DB or snapshot mode)

---

## Project structure

- `main.py` orchestrates the full pipeline
- `config/` contains `settings.yaml`, `subreddits.txt`, `stopwords_custom.txt`
- `reddit/` scraping, filtering, cleaning, export
- `db/` MongoDB access and indexes
- `processing/` embeddings and weights
- `modeling/` UMAP, clustering, c-TF-IDF topics
- `llm/` topic labeling + stance/sentiment batch scripts, JSONL I/O
- `plots/` visual artifacts
- `reports/` CSV reports
- `dashboard/` Next.js frontend

---

## Prerequisites

- Python 3.10+ recommended
- MongoDB (local or remote)
- Node.js + npm (dashboard)
- Optional: GPU for faster embedding/LLM runs

---

## Quick start

1. Python setup:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```
2. MongoDB: set your credentials in `config/settings.yaml`.
3. Environment variables (recommended for secrets):
   ```powershell
   setx REDDIT_CLIENT_ID "..."
   setx REDDIT_CLIENT_SECRET "..."
   setx REDDIT_USER_AGENT "trend-analyzer"
   setx OPENROUTER_API_KEY "..."
   ```
4. Run full pipeline:
   ```powershell
   python main.py
   ```

---

## Configuration (`config/settings.yaml`)

Typical sections:

- `reddit` - credentials and user agent
- `mongodb` - uri and database
- `embeddings` - model, batch size, token limits, overlap
- `openrouter` - API key, model, tokens, timeout
- `reddit_pipeline` - subreddits, filters, comment handling

Many scripts also accept a `--config` flag to point at a local config file.

---

## Subreddit list (`config/subreddits.txt`)

the subreddits to be researched are noted there:

```
science (research)
MachineLearning (ai)
```

---

## Run the pipeline

Full pipeline which extracts data from reddit:

```powershell
python main.py
```

Useful CLI flags:

- `--dry-run` prints the planned commands without executing them
- `--continue-on-error` keeps going even if one step fails
- `--max-input-tokens` limits LLM batch input size
- `--max-output-tokens` caps LLM response size per request

Examples:

```powershell
python main.py --dry-run
python main.py --continue-on-error
python main.py --max-input-tokens 100000 --max-output-tokens 40000
```

---

## Individual steps

running main.py would call the following commands in order:

- `reddit/reddit_to_db.py` fetches posts/comments and stores them in MongoDB
- `processing/embeddings.py` generates embeddings and writes them back to MongoDB
- `plots/plot_embeddings.py` creates an embedding plot and can save projected coords
- `modeling/cluster.py` clusters posts and assigns `topic_id`
- `modeling/cluster_noise.py` post-processes or flags noise points after clustering
- `modeling/ctfidf_topics.py` writes a CSV of top topic terms
- `llm/topic_label_batch.py` labels topics from the CSV inputs and writes results
- `llm/export_missing_stance_sentiment_jsonl.py` exports missing comment labels for batching
- `llm/stance_sentiment_batch.py` labels missing comments for stance and sentiment
- `llm/save_sentiment_stance_db.py` saves stance/sentiment results back to MongoDB
- `processing/weights.py` computes comment weights and post-level aggregates

```powershell
python reddit/reddit_to_db.py --skip-existing --refresh-days 7
python processing/embeddings.py --force
python plots/plot_embeddings.py --plot-output plots/embeddings.png --save-db --umap-cluster-dim 50
python modeling/cluster.py --clusterer hdbscan --save-db
python modeling/cluster_noise.py
python modeling/ctfidf_topics.py --output reports/ctfidf_topics.csv --top-n 12
python llm/topic_label_batch.py --retry-missing 3
python llm/export_missing_stance_sentiment_jsonl.py --output llm/stance_sentiment_missing.jsonl
python llm/stance_sentiment_batch.py --input llm/stance_sentiment_missing.jsonl --output llm/stance_sentiment_results2.jsonl
python llm/save_sentiment_stance_db.py --input llm/stance_sentiment_results2.jsonl --only-missing
python processing/weights.py --save-db
```

---

## MongoDB collections

the DB conteains 2 collections:

`1-posts` (fields you will commonly see):

- `post_id` - Reddit post ID (string)
- `subreddit` - subreddit name
- `title` - post title
- `selftext` - original body text
- `cleaned_selftext` - cleaned body text
- `topic_text` - combined text used for embeddings (title + cleaned body)
- `created_utc` - post timestamp (epoch seconds)
- `score` - post score (upvotes minus downvotes)
- `num_comments` - number of comments at fetch time
- `embedding` - vector embedding (list of floats)
- `embedding_model` - embedding model name
- `embedding_dim` - embedding dimensionality
- `topic_id` - cluster label assigned by the clustering step
- `center_distance` - distance to cluster center (for ranking within a topic)
- `topic_name` - LLM topic label (short name)
- `topic_description` - LLM topic label (description)
- `confidence` - LLM topic label confidence (float, optional)
- `stance_dist_weighted` - weighted stance distribution per post
- `sentiment_dist_weighted` - weighted sentiment distribution per post
- `polarization_score` - polarization score (0..1) when enough stance labels exist
- `snapshot_week` - ISO week (YYYY-Wxx) for grouping
- `created_at`, `updated_at` - DB timestamps

`2-comments` (fields you will commonly see):

- `comment_id` - Reddit comment ID (string)
- `post_id` - parent post ID
- `comment_text` - raw comment text
- `comment_text_clean` - cleaned comment text
- `created_utc` - comment timestamp (epoch seconds)
- `upvote_score` - upvotes (fallback to score if missing)
- `sentiment_label` - sentiment classification (positive/negative/neutral)
- `stance_label` - stance classification (agree/disagree/neutral)
- `llm_confidence` - LLM confidence for stance/sentiment
- `llm_comment_exp` - short LLM rationale (if available)
- `weight` - per-comment aggregation weight
- `snapshot_week` - ISO week (YYYY-Wxx) for grouping
- `created_at`, `updated_at` - DB timestamps

---

## Dashboard (Next.js)

generate a live dashboard using:

```powershell
cd dashboard
npm install
npm run dev
```

Snapshot mode (offline demo):

```powershell
npm run snapshot
```

Dashboard environment variables:

- `MONGODB_URI`
- `MONGODB_DB`
- `DASHBOARD_SNAPSHOT_ONLY` (set to `1` for snapshot-only mode)
- `DASHBOARD_TITLE`

---

## Outputs

- `data/raw/reddit_posts_*.json` - raw export
- `plots/embeddings.png` - UMAP/embedding plot (if enabled)
- `reports/ctfidf_topics.csv` - topic keywords per `topic_id`
- `llm/*missing*.jsonl` - inputs for retry/missing batches
- `llm/*results*.jsonl` - LLM results
- MongoDB collections: `posts`, `comments`

---

## Architecture / Data Flow

```mermaid
flowchart LR
  A[Reddit API] --> B[reddit/reddit_to_db.py]
  B --> C[(MongoDB<br/>posts + comments)]

  C --> D[processing/embeddings.py]
  D --> C

  C --> E[plots/plot_embeddings.py]
  E --> P[plots/embeddings.png]

  C --> F[modeling/cluster.py]
  F --> C

  C --> G[modeling/cluster_noise.py]
  G --> C

  C --> H[modeling/ctfidf_topics.py]
  H --> I[reports/ctfidf_topics.csv]

  I --> J[llm/topic_label_batch.py]
  J --> K[reports/topic_label_results.csv]

  C --> L[llm/export_missing_stance_sentiment_jsonl.py]
  L --> M[llm/stance_sentiment_missing.jsonl]

  M --> N[llm/stance_sentiment_batch.py]
  N --> O[llm/stance_sentiment_results.jsonl]

  O --> Q[llm/save_sentiment_stance_db.py]
  Q --> C

  C --> R[processing/weights.py]
  R --> C

  C --> S[dashboard (Next.js)]


```

---

## Limitations

**Scope and data coverage (code-level)**

- The pipeline only ingests Reddit data. It does not validate claims against the open web, news, or academic sources.
- Only the configured subreddits and filters are analyzed, so results reflect that curated slice, not Reddit overall.
- Comments are fetched with limits and filters, so you may miss long-tail or late-arriving discussion if you dont adjust.
- The pipeline is text-only and does not analyze images, videos, links, or embedded media (yet).
- There is no cross-platform comparison (X, YouTube, forums) and no user-level context beyond what Reddit exposes in the fetched data.

**Modeling and labeling**

- Clustering is unsupervised, so topic boundaries may not align with human intuition.
- LLM labels (topic/sentiment/stance) can be wrong due to sarcasm, missing context, or domain ambiguity.
- LLM outputs can vary across runs and model versions unless you lock model and settings.

**Interpretation and evaluation**

- Reddit is not representative of the whole population.
- Upvotes are a noisy proxy for agreement or importance and can be influenced by timing and visibility.
- Polarization is a relative measure; without baselines, the absolute number is hard to interpret.

**What to add to make results more meaningful**

- Human-labeled evaluation sets for topic labels, stance, and sentiment.
- Inter-annotator agreement to confirm that labels are reliable.
- Topic quality metrics (coherence, stability across runs, cluster purity).
- Calibration of LLM confidence and thresholds for labeling/aggregation.
- External validation against trusted sources for key claims or trends.

---

## Troubleshooting

- `ValueError: Reddit API credentials not found`: Set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` or add them to `config/settings.yaml`.
- `MongoDB settings missing` or connection refused: Verify `mongodb.uri` and `mongodb.database` and ensure MongoDB is running.
- `OpenRouter API key missing`: Set `OPENROUTER_API_KEY` or `openrouter.api_key`.
- `umap-learn is required`: Install with `python -m pip install umap-learn`.
- `matplotlib is required for plotting`: Install with `python -m pip install matplotlib`.
- LLM requests fail or time out: Reduce `--max-input-tokens`, `--max-output-tokens`, or batch size; check rate limits.

---

## Roadmap (next steps)

- Define baselines for polarization and stance interpretation.
- Add web scraping to compare Reddit claims with trusted sources.
- Add image + OCR + meme analysis.
- Add time-series per topic (growth, momentum, change over time).
- Extend to multi-platform analysis.
- Add trend prediction and LLM summaries.

---

## License

No license file is included yet.
