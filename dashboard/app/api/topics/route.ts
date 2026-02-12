import { getDb } from "@/lib/mongo";
import { buildPostMatch, parseFilters } from "@/lib/filters";
import { respondWithFallback } from "@/lib/response";

export async function GET(req: Request) {
  const filters = parseFilters(req);

  return respondWithFallback(
    "topics",
    { range: filters.range, subreddit: filters.subreddit, topic: filters.topic },
    async () => {
      const db = await getDb();
      const match = buildPostMatch(filters);
      if (!filters.topic) {
        match.topic_id = { $exists: true, $ne: null };
      }

      const pipeline = [
        { $match: match },
        {
          $group: {
            _id: "$topic_id",
            count: { $sum: 1 },
            topic_name: { $first: "$topic_name" },
            topic_description: { $first: "$topic_description" },
            avg_comments: { $avg: "$num_comments" },
            avg_score: { $avg: "$score" },
            avg_center_distance: { $avg: "$center_distance" },
          },
        },
        {
          $project: {
            _id: 0,
            topic_id: "$_id",
            topic_name: { $ifNull: ["$topic_name", ""] },
            topic_description: { $ifNull: ["$topic_description", ""] },
            count: 1,
            avg_comments: 1,
            avg_score: 1,
            avg_center_distance: 1,
          },
        },
        { $sort: { count: -1 } },
        { $limit: 80 },
      ];

      const topics = await db.collection("posts").aggregate(pipeline).toArray();

      return { topics };
    },
  );
}
