import type { Filters } from "./types";

const RANGE_DAYS: Record<string, number> = {
  "7d": 7,
  "30d": 30,
  "90d": 90,
  "180d": 180,
  "365d": 365,
};

export type ParsedFilters = Filters & {
  since?: number;
  days?: number;
};

export function parseFilters(req: Request): ParsedFilters {
  const url = new URL(req.url);
  const range = url.searchParams.get("range") || "90d";
  const subredditParam = url.searchParams.get("subreddit") || "";
  const topicParam = url.searchParams.get("topic") || "";
  const subreddit = subredditParam && subredditParam !== "all" ? subredditParam : null;
  const topic = topicParam && topicParam !== "all" ? topicParam : null;
  const days = RANGE_DAYS[range];
  const since = days ? Math.floor((Date.now() - days * 86400000) / 1000) : undefined;

  return { range, subreddit, topic, since, days };
}

export function buildPostMatch(filters: ParsedFilters) {
  const match: Record<string, unknown> = {};
  if (filters.since) {
    match.created_utc = { $gte: filters.since };
  }
  if (filters.subreddit) {
    match.subreddit = filters.subreddit;
  }
  if (filters.topic) {
    match.topic_id = filters.topic;
  }
  return match;
}

export function buildCommentPipeline(filters: ParsedFilters) {
  const pipeline: Record<string, unknown>[] = [];
  const match: Record<string, unknown> = {};
  if (filters.since) {
    match.created_utc = { $gte: filters.since };
  }
  if (Object.keys(match).length > 0) {
    pipeline.push({ $match: match });
  }

  if (filters.subreddit || filters.topic) {
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
    pipeline.push({ $match: postMatch });
  }

  return pipeline;
}

export function getBucketUnit(filters: ParsedFilters): "day" | "week" {
  if (!filters.days) {
    return "week";
  }
  return filters.days > 120 ? "week" : "day";
}
