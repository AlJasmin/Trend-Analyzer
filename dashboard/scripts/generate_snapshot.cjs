const fs = require("fs");
const path = require("path");
const yaml = require("yaml");
const { MongoClient } = require("mongodb");

const RANGE_DAYS = {
  "7d": 7,
  "30d": 30,
  "90d": 90,
  "180d": 180,
  "365d": 365,
};

function parseArgs() {
  const args = process.argv.slice(2);
  const out = { range: "90d", subreddit: null, topic: null };
  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === "--range" && args[i + 1]) {
      out.range = args[i + 1];
      i += 1;
    }
    if (args[i] === "--subreddit" && args[i + 1]) {
      out.subreddit = args[i + 1];
      i += 1;
    }
    if (args[i] === "--topic" && args[i + 1]) {
      out.topic = args[i + 1];
      i += 1;
    }
  }
  return out;
}

function loadConfig() {
  const envUri = process.env.MONGODB_URI;
  const envDb = process.env.MONGODB_DB;
  if (envUri && envDb) {
    return { uri: envUri, database: envDb };
  }
  const configPath = path.join(__dirname, "..", "..", "config", "settings.yaml");
  const raw = fs.readFileSync(configPath, "utf-8");
  const parsed = yaml.parse(raw) || {};
  const mongo = parsed.mongodb || {};
  const uri = envUri || mongo.uri;
  const database = envDb || mongo.database;
  if (!uri || !database) {
    throw new Error("MongoDB settings missing (mongodb.uri / mongodb.database).");
  }
  return { uri, database };
}

function buildPostMatch(filters) {
  const match = {};
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

function buildCommentPipeline(filters) {
  const pipeline = [];
  const match = {};
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
    const postMatch = {};
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

function getBucketUnit(filters) {
  if (!filters.days) {
    return "week";
  }
  return filters.days > 120 ? "week" : "day";
}

function withWeightedDistributions(match) {
  return {
    ...match,
    stance_dist_weighted: { $ne: null },
    sentiment_dist_weighted: { $ne: null },
  };
}

function pickDominant(agree, disagree, neutral) {
  if (agree >= disagree && agree >= neutral) {
    return "agree";
  }
  if (disagree >= agree && disagree >= neutral) {
    return "disagree";
  }
  return "neutral";
}

async function main() {
  const args = parseArgs();
  const days = RANGE_DAYS[args.range];
  const since = days ? Math.floor((Date.now() - days * 86400000) / 1000) : undefined;
  const filters = {
    range: args.range,
    subreddit: args.subreddit,
    topic: args.topic,
    days,
    since,
  };

  const config = loadConfig();
  const client = new MongoClient(config.uri);
  await client.connect();
  const db = client.db(config.database);

  const postsMatch = buildPostMatch(filters);
  const postPipeline = [];
  if (Object.keys(postsMatch).length > 0) {
    postPipeline.push({ $match: postsMatch });
  }
  postPipeline.push({
    $group: {
      _id: null,
      posts: { $sum: 1 },
      avg_comments: { $avg: "$num_comments" },
      avg_post_upvotes: { $avg: "$score" },
    },
  });

  const commentPipeline = buildCommentPipeline(filters);
  const normalizedText = {
    $toLower: { $trim: { input: { $ifNull: ["$comment_text", ""] } } },
  };
  const hasStance = {
    $and: [{ $ne: ["$stance_label", null] }, { $ne: ["$stance_label", ""] }],
  };
  const hasSentiment = {
    $and: [{ $ne: ["$sentiment_label", null] }, { $ne: ["$sentiment_label", ""] }],
  };
  const isLabeled = { $and: [hasStance, hasSentiment] };
  const isDeleted = {
    $in: [normalizedText, ["[deleted]", "[removed]", ""]],
  };

  commentPipeline.push({
    $project: {
      comment_text: 1,
      stance_label: 1,
      sentiment_label: 1,
      llm_confidence: 1,
    },
  });
  commentPipeline.push({
    $group: {
      _id: null,
      comments: { $sum: 1 },
      labeled: { $sum: { $cond: [isLabeled, 1, 0] } },
      deleted: { $sum: { $cond: [isDeleted, 1, 0] } },
      low_confidence: {
        $sum: {
          $cond: [
            { $and: [isLabeled, { $lt: ["$llm_confidence", 0.5] }] },
            1,
            0,
          ],
        },
      },
      confidence_sum: {
        $sum: {
          $cond: [
            { $and: [isLabeled, { $ne: ["$llm_confidence", null] }] },
            "$llm_confidence",
            0,
          ],
        },
      },
      confidence_count: {
        $sum: {
          $cond: [
            { $and: [isLabeled, { $ne: ["$llm_confidence", null] }] },
            1,
            0,
          ],
        },
      },
    },
  });

  const [postStats] = await db.collection("posts").aggregate(postPipeline).toArray();
  const [commentStats] = await db.collection("comments").aggregate(commentPipeline).toArray();
  const commentUpvotesPipeline = buildCommentPipeline(filters);
  commentUpvotesPipeline.push({
    $project: {
      post_id: 1,
      score: { $ifNull: ["$score", { $ifNull: ["$upvote_score", 0] }] },
    },
  });
  commentUpvotesPipeline.push({
    $group: {
      _id: "$post_id",
      comment_upvotes_sum: { $sum: "$score" },
    },
  });
  commentUpvotesPipeline.push({
    $group: {
      _id: null,
      avg_comment_upvotes_per_post: { $avg: "$comment_upvotes_sum" },
    },
  });
  const [commentUpvotes] = await db
    .collection("comments")
    .aggregate(commentUpvotesPipeline)
    .toArray();

  const overview = {
    posts: postStats?.posts || 0,
    comments: commentStats?.comments || 0,
    labeled_comments: commentStats?.labeled || 0,
    label_coverage: commentStats?.comments
      ? commentStats.labeled / commentStats.comments
      : 0,
    avg_confidence:
      commentStats?.confidence_count > 0
        ? commentStats.confidence_sum / commentStats.confidence_count
        : null,
    avg_comments_per_post: postStats?.avg_comments ?? null,
    avg_post_upvotes: postStats?.avg_post_upvotes ?? null,
    avg_comment_upvotes_per_post: commentUpvotes?.avg_comment_upvotes_per_post ?? null,
    deleted_comments: commentStats?.deleted || 0,
    low_confidence_comments: commentStats?.low_confidence || 0,
  };

  const stancePipeline = buildCommentPipeline(filters);
  stancePipeline.push({
    $project: {
      stance: { $ifNull: ["$stance_label", "unknown"] },
      sentiment: { $ifNull: ["$sentiment_label", "unknown"] },
    },
  });
  stancePipeline.push({
    $group: {
      _id: { stance: "$stance", sentiment: "$sentiment" },
      count: { $sum: 1 },
    },
  });
  const stanceRows = await db.collection("comments").aggregate(stancePipeline).toArray();
  const stance_counts = {};
  const sentiment_counts = {};
  const matrix = stanceRows.map((row) => {
    const stance = row._id.stance;
    const sentiment = row._id.sentiment;
    const count = row.count;
    stance_counts[stance] = (stance_counts[stance] || 0) + count;
    sentiment_counts[sentiment] = (sentiment_counts[sentiment] || 0) + count;
    return { stance, sentiment, count };
  });

  const unit = getBucketUnit(filters);
  const buildTrendPipeline = (field) => {
    const pipeline = buildCommentPipeline(filters);
    pipeline.push({ $match: { created_utc: { $gt: 0 } } });
    pipeline.push({
      $addFields: {
        created_at: { $toDate: { $multiply: ["$created_utc", 1000] } },
      },
    });
    pipeline.push({
      $addFields: { bucket: { $dateTrunc: { date: "$created_at", unit } } },
    });
    pipeline.push({
      $project: { bucket: 1, label: { $ifNull: [`$${field}`, "unknown"] } },
    });
    pipeline.push({
      $group: { _id: { bucket: "$bucket", label: "$label" }, count: { $sum: 1 } },
    });
    pipeline.push({ $sort: { "_id.bucket": 1 } });
    return pipeline;
  };

  const buildSeries = (rows) => {
    const map = new Map();
    rows.forEach((row) => {
      const key = row._id.bucket.toISOString();
      if (!map.has(key)) {
        map.set(key, {});
      }
      map.get(key)[row._id.label] = row.count;
    });
    return Array.from(map.entries())
      .sort((a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime())
      .map(([date, counts]) => ({ date, counts }));
  };

  const [stanceTrendRows, sentimentTrendRows] = await Promise.all([
    db.collection("comments").aggregate(buildTrendPipeline("stance_label")).toArray(),
    db.collection("comments").aggregate(buildTrendPipeline("sentiment_label")).toArray(),
  ]);

  const trends = {
    bucket: unit,
    stance_series: buildSeries(stanceTrendRows),
    sentiment_series: buildSeries(sentimentTrendRows),
  };

  const topicsMatch = buildPostMatch(filters);
  if (!filters.topic) {
    topicsMatch.topic_id = { $exists: true, $ne: null };
  }
  const topicsPipeline = [
    { $match: topicsMatch },
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
  const topics = await db.collection("posts").aggregate(topicsPipeline).toArray();

  const subredditPipeline = [
    { $match: buildPostMatch(filters) },
    { $group: { _id: "$subreddit", count: { $sum: 1 } } },
    { $sort: { count: -1 } },
    { $limit: 60 },
    { $project: { _id: 0, subreddit: "$_id", count: 1 } },
  ];
  const subreddits = await db
    .collection("posts")
    .aggregate(subredditPipeline)
    .toArray();

  const opinionMapMatch = withWeightedDistributions(buildPostMatch(filters));
  const opinionMapItems = await db
    .collection("posts")
    .aggregate([
      { $match: opinionMapMatch },
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
      { $limit: 1500 },
    ])
    .toArray();

  const umapMatch = buildPostMatch(filters);
  umapMatch.umap_x = { $ne: null };
  umapMatch.umap_y = { $ne: null };
  const umapPoints = await db
    .collection("posts")
    .aggregate([
      { $match: umapMatch },
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
      { $limit: 2000 },
    ])
    .toArray();

  const topicInsightsMatch = withWeightedDistributions(buildPostMatch(filters));
  if (!filters.topic) {
    topicInsightsMatch.topic_id = { $exists: true, $ne: null };
  }
  const topicInsights = await db
    .collection("posts")
    .aggregate([
      { $match: topicInsightsMatch },
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
      { $limit: 40 },
    ])
    .toArray();

  const sankeyPipeline = buildCommentPipeline(filters);
  sankeyPipeline.push({
    $project: {
      stance: { $ifNull: ["$stance_label", "unknown"] },
      sentiment: { $ifNull: ["$sentiment_label", "unknown"] },
      weight: { $ifNull: ["$weight", 1] },
    },
  });
  sankeyPipeline.push({
    $addFields: {
      stance: { $cond: [{ $eq: ["$stance", "unclear"] }, "unknown", "$stance"] },
    },
  });
  sankeyPipeline.push({
    $group: {
      _id: { sentiment: "$sentiment", stance: "$stance" },
      value: { $sum: "$weight" },
    },
  });
  const sankeyRows = await db.collection("comments").aggregate(sankeyPipeline).toArray();
  let sankey = { facet: "topic", facets: [] };
  if (sankeyRows.length > 0) {
    const links = sankeyRows.map((row) => ({
      source: row._id.sentiment,
      target: row._id.stance,
      value: row.value,
    }));
    const nodes = Array.from(
      new Set(links.flatMap((link) => [link.source, link.target])),
    ).map((name) => ({ name }));
    sankey = {
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
  }

  const timelinePipeline = [];
  if (filters.since) {
    timelinePipeline.push({ $match: { created_utc: { $gte: filters.since } } });
  }
  timelinePipeline.push({
    $lookup: {
      from: "posts",
      localField: "post_id",
      foreignField: "post_id",
      as: "post",
    },
  });
  timelinePipeline.push({ $unwind: "$post" });
  const timelineMatch = {};
  if (filters.subreddit) {
    timelineMatch["post.subreddit"] = filters.subreddit;
  }
  if (filters.topic) {
    timelineMatch["post.topic_id"] = filters.topic;
  }
  if (Object.keys(timelineMatch).length > 0) {
    timelinePipeline.push({ $match: timelineMatch });
  }
  timelinePipeline.push({
    $project: {
      post_id: 1,
      snapshot_week: "$post.snapshot_week",
      polarization_score: "$post.polarization_score",
      sentiment_dist: "$post.sentiment_dist_weighted",
      weight: { $ifNull: ["$weight", 1] },
    },
  });
  timelinePipeline.push({ $match: { snapshot_week: { $ne: null } } });
  timelinePipeline.push({
    $group: {
      _id: { post_id: "$post_id", week: "$snapshot_week" },
      weight_sum: { $sum: "$weight" },
      polarization_score: { $first: "$polarization_score" },
      sentiment_dist: { $first: "$sentiment_dist" },
    },
  });
  timelinePipeline.push({
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
  timelinePipeline.push({
    $group: {
      _id: "$_id.week",
      weight: { $sum: "$weight_sum" },
      polarization_sum: { $sum: { $multiply: ["$polarization_score", "$weight_sum"] } },
      sentiment_sum: { $sum: { $multiply: ["$sentiment_balance", "$weight_sum"] } },
    },
  });
  timelinePipeline.push({
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
  timelinePipeline.push({ $sort: { week: 1 } });
  const timelinePoints = await db.collection("comments").aggregate(timelinePipeline).toArray();

  const leaderboardPipeline = [];
  if (filters.since) {
    leaderboardPipeline.push({ $match: { created_utc: { $gte: filters.since } } });
  }
  leaderboardPipeline.push({
    $lookup: {
      from: "posts",
      localField: "post_id",
      foreignField: "post_id",
      as: "post",
    },
  });
  leaderboardPipeline.push({ $unwind: "$post" });
  const leaderboardMatch = {};
  if (filters.subreddit) {
    leaderboardMatch["post.subreddit"] = filters.subreddit;
  }
  if (filters.topic) {
    leaderboardMatch["post.topic_id"] = filters.topic;
  }
  if (Object.keys(leaderboardMatch).length > 0) {
    leaderboardPipeline.push({ $match: leaderboardMatch });
  }
  leaderboardPipeline.push({
    $project: {
      post_id: 1,
      title: "$post.title",
      subreddit: "$post.subreddit",
      topic_id: "$post.topic_id",
      topic_name: "$post.topic_name",
      polarization_score: "$post.polarization_score",
      weight: { $ifNull: ["$weight", 1] },
    },
  });
  leaderboardPipeline.push({
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
  leaderboardPipeline.push({
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
  leaderboardPipeline.push({ $sort: { comment_weight_sum: -1 } });
  leaderboardPipeline.push({ $limit: 30 });
  const leaderboardItems = await db
    .collection("comments")
    .aggregate(leaderboardPipeline)
    .toArray();

  const heatmapMatch = buildPostMatch(filters);
  if (!filters.topic) {
    heatmapMatch.topic_id = { $exists: true, $ne: null };
  }
  const heatmapTopics = await db
    .collection("posts")
    .aggregate([
      { $match: heatmapMatch },
      {
        $group: {
          _id: "$topic_id",
          topic_name: { $first: "$topic_name" },
          total_comments: { $sum: { $ifNull: ["$num_comments", 0] } },
        },
      },
      { $sort: { total_comments: -1 } },
      { $limit: 12 },
    ])
    .toArray();
  const heatmapTopicIds = heatmapTopics.map((item) => item._id).filter(Boolean);
  let heatmapWeeks = [];
  let heatmapCells = [];
  if (heatmapTopicIds.length > 0) {
    const heatmapRows = await db
      .collection("posts")
      .aggregate([
        { $match: { ...buildPostMatch(filters), topic_id: { $in: heatmapTopicIds } } },
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
      ])
      .toArray();
    heatmapWeeks = Array.from(new Set(heatmapRows.map((row) => row.week))).sort();
    heatmapCells = heatmapRows.map((row) => ({
      topic_id: row.topic_id,
      week: row.week,
      polarization_score: row.polarization_score,
      dominant_label: pickDominant(row.agree_sum, row.disagree_sum, row.neutral_sum),
    }));
  }

  const snapshot = {
    generated_at: new Date().toISOString(),
    overview,
    stance_sentiment: { stance_counts, sentiment_counts, matrix },
    trends,
    topics: { topics },
    subreddits: { subreddits: subreddits.filter((item) => item.subreddit) },
    opinion_map: { mode: "post", items: opinionMapItems },
    umap_landscape: { points: umapPoints },
    topic_insights: { topics: topicInsights },
    sankey,
    polarization_timeline: { points: timelinePoints },
    controversy_leaderboard: { mode: "post", items: leaderboardItems },
    topic_heatmap: {
      topics: heatmapTopics.map((topic) => ({
        topic_id: topic._id,
        topic_name: topic.topic_name || "",
      })),
      weeks: heatmapWeeks,
      cells: heatmapCells,
    },
  };

  const dataDir = path.join(__dirname, "..", "data");
  fs.mkdirSync(dataDir, { recursive: true });
  const outputPath = path.join(dataDir, "snapshot.json");
  fs.writeFileSync(outputPath, JSON.stringify(snapshot, null, 2));
  await client.close();
  console.log(`Snapshot saved to ${outputPath}`);
}

main().catch((error) => {
  console.error("Snapshot generation failed:", error);
  process.exit(1);
});
