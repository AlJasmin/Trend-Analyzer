export type Filters = {
  range: string;
  subreddit?: string | null;
  topic?: string | null;
};

export type Meta = {
  source: "mongo" | "snapshot";
  generated_at: string;
  filters: Filters;
};

export type ApiResponse<T> = {
  meta: Meta;
  data: T;
};

export type OverviewData = {
  posts: number;
  comments: number;
  labeled_comments: number;
  label_coverage: number;
  avg_confidence: number | null;
  avg_comments_per_post: number | null;
  avg_post_upvotes: number | null;
  avg_comment_upvotes_per_post: number | null;
  deleted_comments: number;
  low_confidence_comments: number;
};

export type StanceSentimentCell = {
  stance: string;
  sentiment: string;
  count: number;
};

export type StanceSentimentData = {
  stance_counts: Record<string, number>;
  sentiment_counts: Record<string, number>;
  matrix: StanceSentimentCell[];
};

export type TrendPoint = {
  date: string;
  counts: Record<string, number>;
};

export type TrendData = {
  bucket: "day" | "week";
  stance_series: TrendPoint[];
  sentiment_series: TrendPoint[];
};

export type TopicItem = {
  topic_id: string;
  topic_name: string;
  topic_description: string;
  count: number;
  avg_comments: number | null;
  avg_score: number | null;
  avg_center_distance: number | null;
};

export type TopicsData = {
  topics: TopicItem[];
};

export type SubredditItem = {
  subreddit: string;
  count: number;
};

export type SubredditsData = {
  subreddits: SubredditItem[];
};

export type OpinionMapItem = {
  id: string;
  label: string;
  x: number;
  y: number;
  polarization_score: number;
  size: number;
  topic_id?: string | null;
  topic_name?: string | null;
  subreddit?: string | null;
};

export type OpinionMapData = {
  mode: "post" | "topic";
  items: OpinionMapItem[];
};

export type UmapPoint = {
  id: string;
  label: string;
  x: number;
  y: number;
  polarization_score: number;
  size: number;
  stance_balance?: number;
  sentiment_balance?: number;
  topic_id?: string | null;
  topic_name?: string | null;
  subreddit?: string | null;
};

export type UmapData = {
  points: UmapPoint[];
};

export type TopicInsight = {
  topic_id: string;
  topic_name: string;
  total_comments: number;
  polarization_score: number;
  avg_center_distance: number | null;
  stance_dist: Record<string, number>;
  sentiment_dist: Record<string, number>;
};

export type TopicInsightsData = {
  topics: TopicInsight[];
};

export type SankeyNode = {
  name: string;
};

export type SankeyLink = {
  source: string;
  target: string;
  value: number;
};

export type SankeyFacet = {
  key: string;
  label: string;
  nodes: SankeyNode[];
  links: SankeyLink[];
};

export type SankeyData = {
  facet: "topic" | "subreddit";
  facets: SankeyFacet[];
};

export type PolarizationTimelinePoint = {
  week: string;
  polarization_score: number;
  sentiment_balance: number;
  weight: number;
};

export type PolarizationTimelineData = {
  points: PolarizationTimelinePoint[];
};

export type LeaderboardItem = {
  id: string;
  label: string;
  polarization_score: number;
  comment_weight_sum: number;
  topic_id?: string | null;
  topic_name?: string | null;
  subreddit?: string | null;
};

export type LeaderboardData = {
  mode: "post" | "topic";
  items: LeaderboardItem[];
};

export type TopicHeatmapCell = {
  topic_id: string;
  week: string;
  polarization_score: number;
  dominant_label: string;
};

export type TopicHeatmapData = {
  topics: { topic_id: string; topic_name: string }[];
  weeks: string[];
  cells: TopicHeatmapCell[];
};

export type Snapshot = {
  generated_at: string;
  overview: OverviewData;
  stance_sentiment: StanceSentimentData;
  trends: TrendData;
  topics: TopicsData;
  subreddits: SubredditsData;
  opinion_map: OpinionMapData;
  umap_landscape: UmapData;
  topic_insights: TopicInsightsData;
  sankey: SankeyData;
  polarization_timeline: PolarizationTimelineData;
  controversy_leaderboard: LeaderboardData;
  topic_heatmap: TopicHeatmapData;
};
