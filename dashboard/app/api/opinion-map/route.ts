import { getDb } from "@/lib/mongo";
import { buildPostMatch, parseFilters } from "@/lib/filters";
import { respondWithFallback } from "@/lib/response";
import type { OpinionMapData } from "@/lib/types";

const DEFAULT_POST_LIMIT = 1500;
const DEFAULT_TOPIC_LIMIT = 80;

export async function GET(req: Request) {
  const filters = parseFilters(req);
  const url = new URL(req.url);
  const mode = url.searchParams.get("mode") === "topic" ? "topic" : "post";
  const limitParam = url.searchParams.get("limit");
  const limit =
    Number(limitParam) ||
    (mode === "topic" ? DEFAULT_TOPIC_LIMIT : DEFAULT_POST_LIMIT);

  return respondWithFallback<OpinionMapData>(
    "opinion_map",
    { range: filters.range, subreddit: filters.subreddit, topic: filters.topic },
    async () => {
      const db = await getDb();
      const match = buildPostMatch(filters);
      match.stance_dist_weighted = { $ne: null };
      match.sentiment_dist_weighted = { $ne: null };

      if (mode === "topic") {
        if (!filters.topic) {
          match.topic_id = { $exists: true, $ne: null };
        }
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
            },
          },
          {
            $addFields: {
              polarization_score: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  { $divide: ["$polarization_sum", "$total_comments"] },
                  0,
                ],
              },
              stance_balance: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  {
                    $subtract: [
                      { $divide: ["$agree_sum", "$total_comments"] },
                      { $divide: ["$disagree_sum", "$total_comments"] },
                    ],
                  },
                  0,
                ],
              },
              sentiment_balance: {
                $cond: [
                  { $gt: ["$total_comments", 0] },
                  {
                    $subtract: [
                      { $divide: ["$positive_sum", "$total_comments"] },
                      { $divide: ["$negative_sum", "$total_comments"] },
                    ],
                  },
                  0,
                ],
              },
            },
          },
          {
            $project: {
              _id: 0,
              id: "$_id",
              label: { $ifNull: ["$topic_name", "$_id"] },
              topic_id: "$_id",
              topic_name: { $ifNull: ["$topic_name", ""] },
              x: "$stance_balance",
              y: "$sentiment_balance",
              polarization_score: 1,
              size: "$total_comments",
            },
          },
          { $sort: { size: -1 } },
          { $limit: limit },
        ];
        const items = await db.collection("posts").aggregate(pipeline).toArray();
        return { mode, items };
      }

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
            size: { $ifNull: ["$num_comments", 0] },
            polarization_score: { $ifNull: ["$polarization_score", 0] },
            x: {
              $subtract: [
                { $ifNull: ["$stance_dist_weighted.agree", 0] },
                { $ifNull: ["$stance_dist_weighted.disagree", 0] },
              ],
            },
            y: {
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
      const items = await db.collection("posts").aggregate(pipeline).toArray();
      return { mode, items };
    },
  );
}
