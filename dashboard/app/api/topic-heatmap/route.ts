import { getDb } from "@/lib/mongo";
import { buildPostMatch, parseFilters } from "@/lib/filters";
import { respondWithFallback } from "@/lib/response";
import type { TopicHeatmapData } from "@/lib/types";

const DEFAULT_LIMIT = 12;

export async function GET(req: Request) {
  const filters = parseFilters(req);
  const url = new URL(req.url);
  const limit = Number(url.searchParams.get("limit")) || DEFAULT_LIMIT;

  return respondWithFallback<TopicHeatmapData>(
    "topic_heatmap",
    { range: filters.range, subreddit: filters.subreddit, topic: filters.topic },
    async () => {
      const db = await getDb();
      const match = buildPostMatch(filters);
      if (!filters.topic) {
        match.topic_id = { $exists: true, $ne: null };
      }

      const topTopics = await db
        .collection("posts")
        .aggregate([
          { $match: match },
          {
            $group: {
              _id: "$topic_id",
              topic_name: { $first: "$topic_name" },
              total_comments: { $sum: { $ifNull: ["$num_comments", 0] } },
            },
          },
          { $sort: { total_comments: -1 } },
          { $limit: limit },
        ])
        .toArray();

      const topicIds = topTopics.map((item) => item._id).filter(Boolean);
      if (topicIds.length === 0) {
        return { topics: [], weeks: [], cells: [] };
      }

      const pipeline = [
        { $match: { ...match, topic_id: { $in: topicIds } } },
        {
          $group: {
            _id: { topic_id: "$topic_id", week: "$snapshot_week" },
            topic_name: { $first: "$topic_name" },
            total_comments: { $sum: { $ifNull: ["$num_comments", 0] } },
            polarization_sum: {
              $sum: {
                $multiply: [
                  { $ifNull: ["$polarization_score", 0] },
                  { $ifNull: ["$num_comments", 0] },
                ],
              },
            },
            agree_sum: {
              $sum: {
                $multiply: [
                  { $ifNull: ["$stance_dist_weighted.agree", 0] },
                  { $ifNull: ["$num_comments", 0] },
                ],
              },
            },
            disagree_sum: {
              $sum: {
                $multiply: [
                  { $ifNull: ["$stance_dist_weighted.disagree", 0] },
                  { $ifNull: ["$num_comments", 0] },
                ],
              },
            },
            neutral_sum: {
              $sum: {
                $multiply: [
                  { $ifNull: ["$stance_dist_weighted.neutral", 0] },
                  { $ifNull: ["$num_comments", 0] },
                ],
              },
            },
          },
        },
        { $match: { "_id.week": { $ne: null } } },
        {
          $project: {
            _id: 0,
            topic_id: "$_id.topic_id",
            week: "$_id.week",
            topic_name: { $ifNull: ["$topic_name", ""] },
            total_comments: 1,
            polarization_score: {
              $cond: [
                { $gt: ["$total_comments", 0] },
                { $divide: ["$polarization_sum", "$total_comments"] },
                0,
              ],
            },
            agree_sum: 1,
            disagree_sum: 1,
            neutral_sum: 1,
          },
        },
      ];

      const rows = await db.collection("posts").aggregate(pipeline).toArray();

      const weeks = Array.from(new Set(rows.map((row) => row.week))).sort();
      const cells = rows.map((row) => {
        const dominant_label = pickDominant(row.agree_sum, row.disagree_sum, row.neutral_sum);
        return {
          topic_id: row.topic_id,
          week: row.week,
          polarization_score: row.polarization_score,
          dominant_label,
        };
      });

      return {
        topics: topTopics.map((topic) => ({
          topic_id: topic._id,
          topic_name: topic.topic_name || "",
        })),
        weeks,
        cells,
      };
    },
  );
}

function pickDominant(agree: number, disagree: number, neutral: number) {
  if (agree >= disagree && agree >= neutral) {
    return "agree";
  }
  if (disagree >= agree && disagree >= neutral) {
    return "disagree";
  }
  return "neutral";
}
