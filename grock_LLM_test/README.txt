Groq LLM stance + sentiment pipeline (Groq API)

1) Export data from MongoDB (JSONL)
   python grock_LLM_test/export_db_jsonl.py --sample 200 --output grock_LLM_test/sample.jsonl

2) Run Grok API labeling
   set GROQ_API_KEY=sk-...
   node grock_LLM_test/run_grok_llm.js --input grock_LLM_test/sample.jsonl --output grock_LLM_test/results.jsonl
   (if GROQ_API_KEY is missing, it will fall back to openrouter.api_key in settings.yaml)

Common flags:
  --model llama-3.3-70b-versatile
  --url https://api.groq.com/openai/v1/chat/completions
  --api-key sk-...
  --config config/settings.yaml
  --prompt grock_LLM_test/prompt_stance_sentiment.txt
  --max-input-tokens 15000
  --max-output-tokens 512
  --max-pairs-per-call 10
  --retry-max-pairs 5
  --debug-dir grock_LLM_test/debug
  --temperature 0
  --rate-limit-retries 3
  --rate-limit-wait-ms 15000
  --rate-limit-buffer-ms 1000
  --batch-delay-ms 60000
  --batch-delay-every 2

Notes:
- If you see empty/invalid responses, lower --max-pairs-per-call
  or raise --retry-max-pairs to split batches on retry.
- Output is JSONL with: post_id, comment_id, sentiment, stance, confidence, rationale
