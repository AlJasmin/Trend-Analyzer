import { getDb } from "@/lib/mongo";
import { buildPostMatch, parseFilters } from "@/lib/filters";
import { respondWithFallback } from "@/lib/response";
import type { TopicInsightsData } from "@/lib/types";

const DEFAULT_LIMIT = 40;

export async function GET(req: Request) {
  const filters = parseFilters(req);
  const url = new URL(req.url);
  const limit = Number(url.searchParams.get("limit")) || DEFAULT_LIMIT;

  return respondWithFallback<TopicInsightsData>(
    "topic_insights",
    { range: filters.range, subreddit: filters.subreddit, topic: filters.topic },
    async () => {
      const db = await getDb();
      const match = buildPostMatch(filters);
      if (!filters.topic) {
        match.topic_id = { $exists: true, $ne: null };
      }
      match.stance_dist_weighted = { $ne: null };
      match.sentiment_dist_weighted = { $ne: null };

      const pipeline = [
        { $match: match },
        {
          $group: {
            _id: "$topic_id",
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
            center_distance_sum: {
              $sum: {
                $multiply: [
                  { $ifNull: ["$center_distance", 0] },
                  { $ifNull: ["$num_comments", 0] },
                ],
              },
            },
            center_distance_weight: {
              $sum: {
                $cond: [
                  { $ne: ["$center_distance", null] },
                  { $ifNull: ["$num_comments", 0] },
                  0,
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
            partial_sum: {
              $sum: {
                $multiply: [
                  { $ifNull: ["$stance_dist_weighted.partial", 0] },
                  { $ifNull: ["$num_comments", 0] },
                ],
              },
            },
            unknown_sum: {
              $sum: {
                $multiply: [
                  {
                    $add: [
                      { $ifNull: ["$stance_dist_weighted.unknown", 0] },
                      { $ifNull: ["$stance_dist_weighted.unclear", 0] },
                    ],
                  },
                  { $ifNull: ["$num_comments", 0] },
                ],
              },
            },
            positive_sum: {
              $sum: {
                $multiply: [
                  { $ifNull: ["$sentiment_dist_weighted.positive", 0] },
                  { $ifNull: ["$num_comments", 0] },
                ],
              },
            },
            negative_sum: {
              $sum: {
                $multiply: [
                  { $ifNull: ["$sentiment_dist_weighted.negative", 0] },
                  { $ifNull: ["$num_comments", 0] },
                ],
              },
            },
            neutral_sentiment_sum: {
              $sum: {
                $multiply: [
                  { $ifNull: ["$sentiment_dist_weighted.neutral", 0] },
                  { $ifNull: ["$num_comments", 0] },
                ],
              },
            },
            mixed_sentiment_sum: {
              $sum: {
                $multiply: [
                  { $ifNull: ["$sentiment_dist_weighted.mixed", 0] },
                  { $ifNull: ["$num_comments", 0] },
                ],
              },
            },
            unknown_sentiment_sum: {
              $sum: {
                $multiply: [
                  { $ifNull: ["$sentiment_dist_weighted.unknown", 0] },
                  { $ifNull: ["$num_comments", 0] },
                ],
              },
            },
          },
        },
        {
          $project: {
            _id: 0,
            topic_id: "$_id",
            topic_name: { $ifNull: ["$topic_name", ""] },
            total_comments: 1,
            polarization_score: {
              $cond: [
                { $gt: ["$total_comments", 0] },
                { $divide: ["$polarization_sum", "$total_comments"] },
                0,
              ],
            },
            avg_center_distance: {
              $cond: [
                { $gt: ["$center_distance_weight", 0] },
                { $divide: ["$center_distance_sum", "$center_distance_weight"] },
                null,
              ],
            },
            stance_dist: {
              agree: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  { $divide: ["$agree_sum", "$total_comments"] },
                  0,
                ],
              },
              disagree: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  { $divide: ["$disagree_sum", "$total_comments"] },
                  0,
                ],
              },
              neutral: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  { $divide: ["$neutral_sum", "$total_comments"] },
                  0,
                ],
              },
              partial: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  { $divide: ["$partial_sum", "$total_comments"] },
                  0,
                ],
              },
              unknown: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  { $divide: ["$unknown_sum", "$total_comments"] },
                  0,
                ],
              },
            },
            sentiment_dist: {
              positive: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  { $divide: ["$positive_sum", "$total_comments"] },
                  0,
                ],
              },
              negative: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  { $divide: ["$negative_sum", "$total_comments"] },
                  0,
                ],
              },
              neutral: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  { $divide: ["$neutral_sentiment_sum", "$total_comments"] },
                  0,
                ],
              },
              mixed: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  { $divide: ["$mixed_sentiment_sum", "$total_comments"] },
                  0,
                ],
              },
              unknown: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  { $divide: ["$unknown_sentiment_sum", "$total_comments"] },
                  0,
                ],
              },
            },
          },
        },
        { $sort: { polarization_score: -1 } },
        { $limit: limit },
      ];

      const topics = await db.collection("posts").aggregate(pipeline).toArray();
      return { topics };
    },
  );
}
