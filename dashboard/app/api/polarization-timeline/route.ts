import { getDb } from "@/lib/mongo";
import { parseFilters } from "@/lib/filters";
import { respondWithFallback } from "@/lib/response";
import type { PolarizationTimelineData } from "@/lib/types";

export async function GET(req: Request) {
  const filters = parseFilters(req);

  return respondWithFallback<PolarizationTimelineData>(
    "polarization_timeline",
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
          snapshot_week: "$post.snapshot_week",
          polarization_score: "$post.polarization_score",
          sentiment_dist: "$post.sentiment_dist_weighted",
          weight: { $ifNull: ["$weight", 1] },
        },
      });
      pipeline.push({ $match: { snapshot_week: { $ne: null } } });
      pipeline.push({
        $group: {
          _id: { post_id: "$post_id", week: "$snapshot_week" },
          weight_sum: { $sum: "$weight" },
          polarization_score: { $first: "$polarization_score" },
          sentiment_dist: { $first: "$sentiment_dist" },
        },
      });
      pipeline.push({
        $addFields: {
          sentiment_balance: {
            $subtract: [
              { $ifNull: ["$sentiment_dist.positive", 0] },
              { $ifNull: ["$sentiment_dist.negative", 0] },
            ],
          },
          polarization_score: { $ifNull: ["$polarization_score", 0] },
        },
      });
      pipeline.push({
        $group: {
          _id: "$_id.week",
          weight: { $sum: "$weight_sum" },
          polarization_sum: {
            $sum: { $multiply: ["$polarization_score", "$weight_sum"] },
          },
          sentiment_sum: {
            $sum: { $multiply: ["$sentiment_balance", "$weight_sum"] },
          },
        },
      });
      pipeline.push({
        $project: {
          _id: 0,
          week: "$_id",
          weight: 1,
          polarization_score: {
            $cond: [
              { $gt: ["$weight", 0] },
              { $divide: ["$polarization_sum", "$weight"] },
              0,
            ],
          },
          sentiment_balance: {
            $cond: [
              { $gt: ["$weight", 0] },
              { $divide: ["$sentiment_sum", "$weight"] },
              0,
            ],
          },
        },
      });
      pipeline.push({ $sort: { week: 1 } });

      const points = await db.collection("comments").aggregate(pipeline).toArray();
      return { points };
    },
  );
}
