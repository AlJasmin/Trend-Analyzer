const fs = require("fs");
const path = require("path");

const REPO_ROOT = path.resolve(__dirname, "..");

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = next;
      i += 1;
    }
  }
  return args;
}

function estimateTokens(text) {
  if (!text) return 0;
  return Math.max(1, Math.ceil(text.length / 4));
}

function loadPromptTemplate(promptPath) {
  const raw = fs.readFileSync(promptPath, "utf-8").trim();
  if (!raw.includes("{{pairs_json}}")) {
    throw new Error("Prompt template must include {{pairs_json}} placeholder.");
  }
  return raw;
}

function loadPairs(inputPath) {
  const raw = fs.readFileSync(inputPath, "utf-8").trim();
  if (!raw) return [];
  if (inputPath.toLowerCase().endsWith(".jsonl")) {
    return raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  }
  return JSON.parse(raw);
}

function buildPrompt(template, pairs) {
  const json = JSON.stringify(pairs, null, 2);
  return template.replace("{{pairs_json}}", json);
}

function buildBatches(pairs, template, maxInputTokens, maxPairsPerCall) {
  const batches = [];
  let current = [];
  for (const pair of pairs) {
    const candidate = current.concat(pair);
    if (maxPairsPerCall && candidate.length > maxPairsPerCall) {
      if (current.length) {
        batches.push(current);
        current = [];
      }
    }
    if (maxInputTokens > 0) {
      const prompt = buildPrompt(template, candidate.length ? candidate : [pair]);
      const tokens = estimateTokens(prompt);
      if (tokens > maxInputTokens) {
        if (current.length) {
          batches.push(current);
          current = [pair];
        } else {
          console.warn("Skipping pair (too large):", pair.comment_id || pair.post_id);
        }
        continue;
      }
    }
    current = candidate.length ? candidate : [pair];
  }
  if (current.length) {
    batches.push(current);
  }
  return batches;
}

function chunkArray(items, size) {
  const out = [];
  for (let i = 0; i < items.length; i += size) {
    out.push(items.slice(i, i + size));
  }
  return out;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseRetryDelayMs(message) {
  if (!message) return null;
  const match = message.match(/try again in ([0-9.]+)s/i);
  if (!match) return null;
  const seconds = Number(match[1]);
  if (!Number.isFinite(seconds)) return null;
  return Math.ceil(seconds * 1000);
}

async function callGrokChat({
  url,
  apiKey,
  model,
  prompt,
  temperature,
  maxOutputTokens,
  jsonMode,
  timeoutMs,
}) {
  const payload = {
    model,
    messages: [
      { role: "system", content: "You are a precise JSON classifier." },
      { role: "user", content: prompt },
    ],
    temperature,
  };
  if (maxOutputTokens) {
    payload.max_tokens = Number(maxOutputTokens);
  }
  if (jsonMode) {
    payload.response_format = { type: "json_object" };
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  const data = await res.json();
  const message = data && data.choices && data.choices[0] && data.choices[0].message;
  const content = message && message.content ? message.content : "";
  return content || "";
}

async function callWithRateLimitRetry(params, retries, waitMs, bufferMs) {
  let lastError = null;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      return await callGrokChat(params);
    } catch (err) {
      lastError = err;
      const msg = err && err.message ? err.message : String(err);
      const isRateLimit = msg.includes("HTTP 429");
      if (!isRateLimit || attempt === retries) {
        throw err;
      }
      const retryDelay = parseRetryDelayMs(msg);
      const delay = (retryDelay || waitMs) + bufferMs;
      console.warn(
        `Rate limit hit. Waiting ${Math.ceil(delay / 1000)}s before retry ${attempt + 1}/${retries}.`
      );
      await sleep(delay);
    }
  }
  throw lastError;
}

function loadOpenrouterKey(configPath) {
  try {
    const raw = fs.readFileSync(configPath, "utf-8");
    const lines = raw.split(/\r?\n/);
    let inSection = false;
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      if (/^openrouter\s*:/i.test(trimmed)) {
        inSection = true;
        continue;
      }
      if (!line.startsWith(" ") && !line.startsWith("\t")) {
        inSection = false;
      }
      if (!inSection) continue;
      const match = trimmed.match(/^api_key\s*:\s*["']?(.+?)["']?\s*$/i);
      if (match && match[1]) {
        return match[1].trim();
      }
    }
  } catch (err) {
    return "";
  }
  return "";
}

function parseJsonArray(text) {
  const trimmed = (text || "").trim();
  if (!trimmed) return [];
  try {
    const data = JSON.parse(trimmed);
    if (Array.isArray(data)) return data;
    if (data && Array.isArray(data.items)) return data.items;
    return [];
  } catch (err) {
    const start = trimmed.indexOf("[");
    const end = trimmed.lastIndexOf("]");
    if (start !== -1 && end !== -1 && end > start) {
      try {
        return JSON.parse(trimmed.slice(start, end + 1));
      } catch (err2) {
        return [];
      }
    }
    return [];
  }
}

function writeJsonl(outputPath, rows, append) {
  const flags = append ? "a" : "w";
  const lines = rows.map((row) => JSON.stringify(row));
  fs.writeFileSync(outputPath, lines.join("\n") + "\n", { encoding: "utf-8", flag: flags });
}

async function main() {
  const args = parseArgs(process.argv);
  const model = args.model || "llama-3.3-70b-versatile";
  const url = args.url || "https://api.groq.com/openai/v1/chat/completions";
  const configPath = args.config || path.join(REPO_ROOT, "config", "settings.yaml");
  const apiKey =
    args["api-key"] ||
    process.env.GROQ_API_KEY ||
    process.env.GROK_API_KEY ||
    loadOpenrouterKey(configPath);
  if (!apiKey) {
    console.error("Missing API key. Set --api-key, GROQ_API_KEY, or openrouter.api_key in settings.yaml.");
    process.exit(1);
  }
  const promptPath = args.prompt || path.join(__dirname, "prompt_stance_sentiment.txt");
  const maxInputTokens = Number(args["max-input-tokens"] || 15000);
  const maxOutputTokens = args["max-output-tokens"] ? Number(args["max-output-tokens"]) : null;
  const temperature = args.temperature ? Number(args.temperature) : 0;
  const maxPairsPerCall = args["max-pairs-per-call"]
    ? Number(args["max-pairs-per-call"])
    : 0;
  const outputPath = args.output || path.join(__dirname, "results.jsonl");
  const dryRun = Boolean(args["dry-run"]);
  const append = Boolean(args.append);
  const debugDir = args["debug-dir"] ? args["debug-dir"] : "";
  const retryMaxPairs = args["retry-max-pairs"] ? Number(args["retry-max-pairs"]) : 0;
  const jsonMode = args["json-mode"] ? true : false;
  const timeoutMs = args["timeout-ms"] ? Number(args["timeout-ms"]) : 60000;
  const batchDelayMs = args["batch-delay-ms"] ? Number(args["batch-delay-ms"]) : 0;
  const batchDelayEvery = args["batch-delay-every"]
    ? Number(args["batch-delay-every"])
    : 0;
  const rateLimitRetries = args["rate-limit-retries"]
    ? Number(args["rate-limit-retries"])
    : 3;
  const rateLimitWaitMs = args["rate-limit-wait-ms"] ? Number(args["rate-limit-wait-ms"]) : 15000;
  const rateLimitBufferMs = args["rate-limit-buffer-ms"]
    ? Number(args["rate-limit-buffer-ms"])
    : 1000;

  let pairs = [];
  if (args.input) {
    pairs = loadPairs(args.input);
  } else if (args.post && args.comment) {
    pairs = [
      {
        post_id: "post_1",
        comment_id: "comment_1",
        post_text: args.post,
        comment_text: args.comment,
      },
    ];
  } else {
    console.error("Provide --input or --post + --comment");
    process.exit(1);
  }

  const template = loadPromptTemplate(promptPath);
  const batches = buildBatches(pairs, template, maxInputTokens, maxPairsPerCall);
  console.log(`Prepared ${batches.length} batches from ${pairs.length} pairs.`);

  if (dryRun) {
    batches.forEach((batch, idx) => {
      const prompt = buildPrompt(template, batch);
      const tokenNote = maxInputTokens > 0 ? ` tokens=${estimateTokens(prompt)}` : "";
      console.log(`Batch ${idx + 1}: pairs=${batch.length}${tokenNote}`);
    });
    return;
  }

  if (debugDir) {
    fs.mkdirSync(debugDir, { recursive: true });
  }

  let batchIndex = 0;
  for (let i = 0; i < batches.length; i += 1) {
    const batch = batches[i];
    const prompt = buildPrompt(template, batch);
    const response = await callWithRateLimitRetry(
      {
        url,
        apiKey,
        model,
        prompt,
        temperature,
        maxOutputTokens,
        jsonMode,
        timeoutMs,
      },
      rateLimitRetries,
      rateLimitWaitMs,
      rateLimitBufferMs
    );
    batchIndex += 1;

    if (debugDir) {
      const debugPath = path.join(debugDir, `batch_${batchIndex}.txt`);
      fs.writeFileSync(debugPath, response || "", "utf-8");
    }

    let parsed = parseJsonArray(response);
    if (!parsed.length && retryMaxPairs && batch.length > retryMaxPairs) {
      const smaller = chunkArray(batch, retryMaxPairs);
      console.warn(
        `Empty/invalid response for batch ${batchIndex}. Retrying in ${smaller.length} smaller batches.`
      );
      for (let j = 0; j < smaller.length; j += 1) {
        const sub = smaller[j];
        const subPrompt = buildPrompt(template, sub);
        const subResponse = await callWithRateLimitRetry(
          {
            url,
            apiKey,
            model,
            prompt: subPrompt,
            temperature,
            maxOutputTokens,
            jsonMode,
            timeoutMs,
          },
          rateLimitRetries,
          rateLimitWaitMs,
          rateLimitBufferMs
        );
        batchIndex += 1;
        if (debugDir) {
          const debugPath = path.join(debugDir, `batch_${batchIndex}.txt`);
          fs.writeFileSync(debugPath, subResponse || "", "utf-8");
        }
        const subParsed = parseJsonArray(subResponse);
        if (!subParsed.length) {
          console.warn(`Empty/invalid response for batch ${batchIndex}`);
          continue;
        }
        writeJsonl(outputPath, subParsed, append || batchIndex > 1);
        console.log(`Wrote ${subParsed.length} rows from batch ${batchIndex}`);
        if (batchDelayEvery > 0) {
          if (batchIndex % batchDelayEvery === 0 && batchDelayMs > 0) {
            await sleep(batchDelayMs);
          }
        } else if (batchDelayMs > 0) {
          await sleep(batchDelayMs);
        }
      }
      continue;
    }

    if (!parsed.length) {
      console.warn(`Empty/invalid response for batch ${batchIndex}`);
      continue;
    }

    writeJsonl(outputPath, parsed, append || batchIndex > 1);
    console.log(`Wrote ${parsed.length} rows from batch ${batchIndex}`);
    if (batchDelayEvery > 0) {
      if (batchIndex % batchDelayEvery === 0 && batchDelayMs > 0) {
        await sleep(batchDelayMs);
      }
    } else if (batchDelayMs > 0) {
      await sleep(batchDelayMs);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
