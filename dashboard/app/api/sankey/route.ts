import { getDb } from "@/lib/mongo";
import { buildCommentPipeline, parseFilters } from "@/lib/filters";
import { respondWithFallback } from "@/lib/response";
import type { SankeyData } from "@/lib/types";

const UNCLEAR_LABEL = "unclear";
const UNKNOWN_LABEL = "unknown";

export async function GET(req: Request) {
  const filters = parseFilters(req);

  return respondWithFallback<SankeyData>(
    "sankey",
    { range: filters.range, subreddit: filters.subreddit, topic: filters.topic },
    async () => {
      const db = await getDb();
      const pipeline: Record<string, unknown>[] = buildCommentPipeline(filters);

      pipeline.push({
        $project: {
          stance: { $ifNull: ["$stance_label", UNKNOWN_LABEL] },
          sentiment: { $ifNull: ["$sentiment_label", UNKNOWN_LABEL] },
          weight: { $ifNull: ["$weight", 1] },
        },
      });
      pipeline.push({
        $addFields: {
          stance: {
            $cond: [{ $eq: ["$stance", UNCLEAR_LABEL] }, UNKNOWN_LABEL, "$stance"],
          },
        },
      });
      pipeline.push({
        $group: {
          _id: {
            sentiment: "$sentiment",
            stance: "$stance",
          },
          value: { $sum: "$weight" },
        },
      });

      const rows = await db.collection("comments").aggregate(pipeline).toArray();
      if (rows.length === 0) {
        return { facet: "topic", facets: [] };
      }

      const links = rows.map((row) => ({
        source: row._id.sentiment,
        target: row._id.stance,
        value: row.value,
      }));
      const nodes = buildNodes(links);

      return {
        facet: "topic",
        facets: [
          {
            key: "all",
            label: "Current filters",
            nodes,
            links,
          },
        ],
      };
    },
  );
}

function buildNodes(links: { source: string; target: string }[]) {
  const set = new Set<string>();
  links.forEach((link) => {
    set.add(link.source);
    set.add(link.target);
  });
  return Array.from(set).map((name) => ({ name }));
}
