import { getDb } from "@/lib/mongo";
import { buildPostMatch, parseFilters } from "@/lib/filters";
import { respondWithFallback } from "@/lib/response";

export async function GET(req: Request) {
  const filters = parseFilters(req);

  return respondWithFallback(
    "subreddits",
    { range: filters.range, subreddit: filters.subreddit, topic: filters.topic },
    async () => {
      const db = await getDb();
      const match = buildPostMatch(filters);
      const pipeline = [
        { $match: match },
        {
          $group: {
            _id: "$subreddit",
            count: { $sum: 1 },
          },
        },
        { $sort: { count: -1 } },
        { $limit: 60 },
        {
          $project: {
            _id: 0,
            subreddit: "$_id",
            count: 1,
          },
        },
      ];

      const subreddits = await db.collection("posts").aggregate(pipeline).toArray();
      return { subreddits: subreddits.filter((item) => item.subreddit) };
    },
  );
}
