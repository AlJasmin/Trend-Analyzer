Local LLM stance + sentiment test (Ollama HTTP API)

1) Install and run Ollama
   https://ollama.com

2) Pull a model (examples)
   ollama pull llama3:8b
   ollama pull qwen2.5:7b-instruct

3) Prepare input (JSONL or JSON)
   Each row/object:
   {
     "post_id": "p1",
     "comment_id": "c1",
     "post_text": "Post text here",
     "comment_text": "Comment text here"
   }

4) Run (batch input)
   node local_LLM_test/run_local_llm.js --input local_LLM_test/sample.jsonl --output local_LLM_test/results.jsonl

5) Run (single pair)
   node local_LLM_test/run_local_llm.js --post "Post text" --comment "Comment text"

Common flags:
  --model llama3:8b
  --url http://localhost:11434/api/chat
  --prompt local_LLM_test/prompt_stance_sentiment.txt
  --max-input-tokens 8000
  --max-output-tokens 512
  --max-pairs-per-call 20
  --temperature 0.2
  --dry-run

Output:
  JSONL with fields: post_id, comment_id, sentiment, stance, confidence, rationale
