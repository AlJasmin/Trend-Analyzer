import { getDb } from "@/lib/mongo";
import { parseFilters } from "@/lib/filters";
import { respondWithFallback } from "@/lib/response";
import type { LeaderboardData } from "@/lib/types";

const DEFAULT_LIMIT = 30;

export async function GET(req: Request) {
  const filters = parseFilters(req);
  const url = new URL(req.url);
  const mode = url.searchParams.get("mode") === "topic" ? "topic" : "post";
  const limit = Number(url.searchParams.get("limit")) || DEFAULT_LIMIT;

  return respondWithFallback<LeaderboardData>(
    "controversy_leaderboard",
    { range: filters.range, subreddit: filters.subreddit, topic: filters.topic },
    async () => {
      const db = await getDb();
      const pipeline: Record<string, unknown>[] = [];

      if (filters.since) {
        pipeline.push({ $match: { created_utc: { $gte: filters.since } } });
      }

      pipeline.push({
        $lookup: {
          from: "posts",
          localField: "post_id",
          foreignField: "post_id",
          as: "post",
        },
      });
      pipeline.push({ $unwind: "$post" });

      const postMatch: Record<string, unknown> = {};
      if (filters.subreddit) {
        postMatch["post.subreddit"] = filters.subreddit;
      }
      if (filters.topic) {
        postMatch["post.topic_id"] = filters.topic;
      }
      if (Object.keys(postMatch).length > 0) {
        pipeline.push({ $match: postMatch });
      }

      pipeline.push({
        $project: {
          post_id: 1,
          topic_id: "$post.topic_id",
          topic_name: "$post.topic_name",
          subreddit: "$post.subreddit",
          title: "$post.title",
          polarization_score: "$post.polarization_score",
          weight: { $ifNull: ["$weight", 1] },
        },
      });

      if (mode === "topic") {
        pipeline.push({ $match: { topic_id: { $ne: null } } });
        pipeline.push({
          $group: {
            _id: "$topic_id",
            topic_name: { $first: "$topic_name" },
            weight_sum: { $sum: "$weight" },
            polarization_sum: {
              $sum: { $multiply: [{ $ifNull: ["$polarization_score", 0] }, "$weight"] },
            },
          },
        });
        pipeline.push({
          $project: {
            _id: 0,
            id: "$_id",
            label: { $ifNull: ["$topic_name", "$_id"] },
            topic_id: "$_id",
            topic_name: { $ifNull: ["$topic_name", ""] },
            comment_weight_sum: "$weight_sum",
            polarization_score: {
              $cond: [
                { $gt: ["$weight_sum", 0] },
                { $divide: ["$polarization_sum", "$weight_sum"] },
                0,
              ],
            },
          },
        });
        pipeline.push({ $sort: { comment_weight_sum: -1 } });
        pipeline.push({ $limit: limit });
      } else {
        pipeline.push({
          $group: {
            _id: "$post_id",
            title: { $first: "$title" },
            subreddit: { $first: "$subreddit" },
            topic_id: { $first: "$topic_id" },
            topic_name: { $first: "$topic_name" },
            polarization_score: { $first: "$polarization_score" },
            comment_weight_sum: { $sum: "$weight" },
          },
        });
        pipeline.push({
          $project: {
            _id: 0,
            id: "$_id",
            label: { $ifNull: ["$title", "$_id"] },
            subreddit: "$subreddit",
            topic_id: "$topic_id",
            topic_name: { $ifNull: ["$topic_name", ""] },
            polarization_score: { $ifNull: ["$polarization_score", 0] },
            comment_weight_sum: 1,
          },
        });
        pipeline.push({ $sort: { comment_weight_sum: -1 } });
        pipeline.push({ $limit: limit });
      }

      const items = await db.collection("comments").aggregate(pipeline).toArray();
      return { mode, items };
    },
  );
}
