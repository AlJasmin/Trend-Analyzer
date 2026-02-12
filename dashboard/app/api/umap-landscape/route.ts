import { getDb } from "@/lib/mongo";
import { buildPostMatch, parseFilters } from "@/lib/filters";
import { respondWithFallback } from "@/lib/response";
import type { UmapData } from "@/lib/types";

const DEFAULT_LIMIT = 2000;

export async function GET(req: Request) {
  const filters = parseFilters(req);
  const url = new URL(req.url);
  const limit = Number(url.searchParams.get("limit")) || DEFAULT_LIMIT;

  return respondWithFallback<UmapData>(
    "umap_landscape",
    { range: filters.range, subreddit: filters.subreddit, topic: filters.topic },
    async () => {
      const db = await getDb();
      const match = buildPostMatch(filters);
      match.umap_x = { $ne: null };
      match.umap_y = { $ne: null };

      const pipeline = [
        { $match: match },
        {
          $project: {
            _id: 0,
            id: "$post_id",
            label: "$title",
            subreddit: "$subreddit",
            topic_id: "$topic_id",
            topic_name: { $ifNull: ["$topic_name", ""] },
            x: "$umap_x",
            y: "$umap_y",
            size: { $ifNull: ["$num_comments", 0] },
            polarization_score: { $ifNull: ["$polarization_score", 0] },
            stance_balance: {
              $subtract: [
                { $ifNull: ["$stance_dist_weighted.agree", 0] },
                { $ifNull: ["$stance_dist_weighted.disagree", 0] },
              ],
            },
            sentiment_balance: {
              $subtract: [
                { $ifNull: ["$sentiment_dist_weighted.positive", 0] },
                { $ifNull: ["$sentiment_dist_weighted.negative", 0] },
              ],
            },
          },
        },
        { $sort: { size: -1 } },
        { $limit: limit },
      ];

      const points = await db.collection("posts").aggregate(pipeline).toArray();
      return { points };
    },
  );
}
