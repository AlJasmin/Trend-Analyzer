import { getDb } from "@/lib/mongo";
import { buildCommentPipeline, getBucketUnit, parseFilters } from "@/lib/filters";
import { respondWithFallback } from "@/lib/response";
import type { TrendData } from "@/lib/types";

type TrendRow = {
  _id: {
    bucket: Date;
    label: string;
  };
  count: number;
};

const UNCLEAR_LABEL = "unclear";
const UNKNOWN_LABEL = "unknown";

function mergeUnclearIntoUnknown(data: TrendData): TrendData {
  const stance_series = data.stance_series.map((point) => {
    const counts = { ...point.counts };
    const unclear = counts[UNCLEAR_LABEL] ?? 0;
    if (unclear) {
      counts[UNKNOWN_LABEL] = (counts[UNKNOWN_LABEL] ?? 0) + unclear;
      delete counts[UNCLEAR_LABEL];
    }
    return { ...point, counts };
  });

  return { ...data, stance_series };
}

function buildSeries(rows: TrendRow[]) {
  const map = new Map<string, Record<string, number>>();
  rows.forEach((row) => {
    const key = row._id.bucket.toISOString();
    if (!map.has(key)) {
      map.set(key, {});
    }
    map.get(key)![row._id.label] = row.count;
  });

  return Array.from(map.entries())
    .sort((a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime())
    .map(([date, counts]) => ({ date, counts }));
}

export async function GET(req: Request) {
  const filters = parseFilters(req);

  return respondWithFallback(
    "trends",
    { range: filters.range, subreddit: filters.subreddit, topic: filters.topic },
    async () => {
      const db = await getDb();
      const unit = getBucketUnit(filters);

      const buildPipeline = (field: "stance_label" | "sentiment_label") => {
        const pipeline = buildCommentPipeline(filters);
        pipeline.push({ $match: { created_utc: { $gt: 0 } } });
        pipeline.push({
          $addFields: {
            created_at: { $toDate: { $multiply: ["$created_utc", 1000] } },
          },
        });
        pipeline.push({
          $addFields: {
            bucket: { $dateTrunc: { date: "$created_at", unit } },
          },
        });
        pipeline.push({
          $project: {
            bucket: 1,
            label: { $ifNull: [`$${field}`, "unknown"] },
          },
        });
        pipeline.push({
          $group: {
            _id: { bucket: "$bucket", label: "$label" },
            count: { $sum: 1 },
          },
        });
        pipeline.push({ $sort: { "_id.bucket": 1 } });
        return pipeline;
      };

      const [stanceRows, sentimentRows] = await Promise.all([
        db.collection("comments").aggregate(buildPipeline("stance_label")).toArray(),
        db.collection("comments").aggregate(buildPipeline("sentiment_label")).toArray(),
      ]);

      return {
        bucket: unit,
        stance_series: buildSeries(stanceRows as TrendRow[]),
        sentiment_series: buildSeries(sentimentRows as TrendRow[]),
      };
    },
    mergeUnclearIntoUnknown,
  );
}
