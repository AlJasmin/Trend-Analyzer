import { getDb } from "@/lib/mongo";
import { buildCommentPipeline, parseFilters } from "@/lib/filters";
import { respondWithFallback } from "@/lib/response";
import type { StanceSentimentCell, StanceSentimentData } from "@/lib/types";

const UNCLEAR_LABEL = "unclear";
const UNKNOWN_LABEL = "unknown";

function normalizeStanceLabel(label: string) {
  return label === UNCLEAR_LABEL ? UNKNOWN_LABEL : label;
}

function mergeStanceSentiment(data: StanceSentimentData): StanceSentimentData {
  const matrix = data.matrix ?? [];
  const merged = new Map<string, StanceSentimentCell>();
  const stance_counts: Record<string, number> = {};
  const sentiment_counts: Record<string, number> = {};

  for (const cell of matrix) {
    const stance = normalizeStanceLabel(cell.stance);
    const sentiment = cell.sentiment;
    const key = `${stance}::${sentiment}`;
    const existing = merged.get(key);
    if (existing) {
      existing.count += cell.count;
    } else {
      merged.set(key, { stance, sentiment, count: cell.count });
    }
  }

  for (const cell of merged.values()) {
    stance_counts[cell.stance] = (stance_counts[cell.stance] || 0) + cell.count;
    sentiment_counts[cell.sentiment] = (sentiment_counts[cell.sentiment] || 0) + cell.count;
  }

  return {
    ...data,
    matrix: Array.from(merged.values()),
    stance_counts,
    sentiment_counts,
  };
}

export async function GET(req: Request) {
  const filters = parseFilters(req);

  return respondWithFallback(
    "stance_sentiment",
    { range: filters.range, subreddit: filters.subreddit, topic: filters.topic },
    async () => {
      const db = await getDb();
      const pipeline = buildCommentPipeline(filters);
      pipeline.push({
        $project: {
          stance: { $ifNull: ["$stance_label", "unknown"] },
          sentiment: { $ifNull: ["$sentiment_label", "unknown"] },
        },
      });
      pipeline.push({
        $group: {
          _id: { stance: "$stance", sentiment: "$sentiment" },
          count: { $sum: 1 },
        },
      });

      const rows = await db.collection("comments").aggregate(pipeline).toArray();
      const stance_counts: Record<string, number> = {};
      const sentiment_counts: Record<string, number> = {};

      const matrix = rows.map((row) => {
        const stance = row._id.stance;
        const sentiment = row._id.sentiment;
        const count = row.count;
        stance_counts[stance] = (stance_counts[stance] || 0) + count;
        sentiment_counts[sentiment] = (sentiment_counts[sentiment] || 0) + count;
        return { stance, sentiment, count };
      });

      return { stance_counts, sentiment_counts, matrix };
    },
    mergeStanceSentiment,
  );
}
