"use client";

import { useEffect, useMemo, useState } from "react";
import type {
  ApiResponse,
  OverviewData,
  StanceSentimentData,
  TrendData,
  TopicsData,
  SubredditsData,
  TopicItem,
  UmapData,
  TopicInsightsData,
  SankeyData,
  PolarizationTimelineData,
  LeaderboardData,
} from "@/lib/types";
import { formatDateTime, formatNumber, formatPercent } from "@/lib/format";
import FilterBar from "./FilterBar";
import HeatmapChart from "./HeatmapChart";
import KpiCard from "./KpiCard";
import TopicTable from "./TopicTable";
import TrendChart from "./TrendChart";
import OpinionMap from "./OpinionMap";
import UmapLandscape from "./UmapLandscape";
import TopicStackedBars from "./TopicStackedBars";
import SankeyPanel from "./SankeyPanel";
import PolarizationTimeline from "./PolarizationTimeline";
import ControversyLeaderboard from "./ControversyLeaderboard";
import OpinionFingerprint from "./OpinionFingerprint";
import PolarizationCoherence from "./PolarizationCoherence";

const DEFAULT_RANGE = "90d";
const RANGE_LABELS: Record<string, string> = {
  "7d": "Last 7 days",
  "30d": "Last 30 days",
  "90d": "Last 90 days",
  "180d": "Last 180 days",
  "365d": "Last 12 months",
  all: "All time",
};

async function fetchJson<T>(url: string): Promise<ApiResponse<T>> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

export default function Dashboard() {
  const [range, setRange] = useState(DEFAULT_RANGE);
  const [subreddit, setSubreddit] = useState("all");
  const [topic, setTopic] = useState("all");
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [stanceSentiment, setStanceSentiment] = useState<StanceSentimentData | null>(null);
  const [trends, setTrends] = useState<TrendData | null>(null);
  const [topics, setTopics] = useState<TopicsData | null>(null);
  const [subreddits, setSubreddits] = useState<string[]>([]);
  const [topicOptions, setTopicOptions] = useState<TopicItem[]>([]);
  const [insightMode, setInsightMode] = useState<"post" | "topic">("post");
  const [umapMode, setUmapMode] = useState<"sentiment" | "stance">("sentiment");
  const [umapLandscape, setUmapLandscape] = useState<UmapData | null>(null);
  const [umapLandscapeGlobal, setUmapLandscapeGlobal] = useState<UmapData | null>(null);
  const [topicInsights, setTopicInsights] = useState<TopicInsightsData | null>(null);
  const [sankey, setSankey] = useState<SankeyData | null>(null);
  const [timeline, setTimeline] = useState<PolarizationTimelineData | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardData | null>(null);
  const [source, setSource] = useState<"mongo" | "snapshot">("mongo");
  const [updatedAt, setUpdatedAt] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const buildParams = (extra?: Record<string, string>) => {
    const params = new URLSearchParams({ range, ...(extra || {}) });
    if (subreddit && subreddit !== "all") {
      params.set("subreddit", subreddit);
    }
    if (topic && topic !== "all") {
      params.set("topic", topic);
    }
    return params;
  };

  useEffect(() => {
    let isMounted = true;
    const params = buildParams();

    setLoading(true);
    setError(null);

    Promise.all([
      fetchJson<OverviewData>(`/api/overview?${params.toString()}`),
      fetchJson<StanceSentimentData>(`/api/stance-sentiment?${params.toString()}`),
      fetchJson<TrendData>(`/api/trends?${params.toString()}`),
      fetchJson<TopicsData>(`/api/topics?${params.toString()}`),
    ])
      .then(([overviewRes, stanceRes, trendsRes, topicsRes]) => {
        if (!isMounted) {
          return;
        }
        setOverview(overviewRes.data);
        setStanceSentiment(stanceRes.data);
        setTrends(trendsRes.data);
        setTopics(topicsRes.data);
        const useSnapshot =
          overviewRes.meta.source === "snapshot" ||
          stanceRes.meta.source === "snapshot" ||
          trendsRes.meta.source === "snapshot" ||
          topicsRes.meta.source === "snapshot";
        setSource(useSnapshot ? "snapshot" : "mongo");
        setUpdatedAt(overviewRes.meta.generated_at);
      })
      .catch((err) => {
        if (!isMounted) {
          return;
        }
        setError(err instanceof Error ? err.message : "Unknown error");
      })
      .finally(() => {
        if (isMounted) {
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [range, subreddit, topic]);

  useEffect(() => {
    let isMounted = true;

    fetchJson<UmapData>("/api/umap-landscape?range=all")
      .then((response) => {
        if (!isMounted) {
          return;
        }
        setUmapLandscapeGlobal(response.data);
      })
      .catch(() => {
        if (!isMounted) {
          return;
        }
        setUmapLandscapeGlobal(null);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    let isMounted = true;
    const params = buildParams({ mode: insightMode });

    fetchJson<LeaderboardData>(`/api/leaderboard?${params.toString()}`)
      .then((leaderboardRes) => {
        if (!isMounted) {
          return;
        }
        setLeaderboard(leaderboardRes.data);
      })
      .catch(() => {
        if (!isMounted) {
          return;
        }
        setLeaderboard(null);
      });

    return () => {
      isMounted = false;
    };
  }, [range, subreddit, topic, insightMode]);

  useEffect(() => {
    let isMounted = true;
    const params = buildParams();

    Promise.all([
      fetchJson<UmapData>(`/api/umap-landscape?${params.toString()}`),
      fetchJson<TopicInsightsData>(`/api/topic-insights?${params.toString()}`),
      fetchJson<PolarizationTimelineData>(`/api/polarization-timeline?${params.toString()}`),
    ])
      .then(([umapRes, insightsRes, timelineRes]) => {
        if (!isMounted) {
          return;
        }
        setUmapLandscape(umapRes.data);
        setTopicInsights(insightsRes.data);
        setTimeline(timelineRes.data);
      })
      .catch(() => {
        if (!isMounted) {
          return;
        }
        setUmapLandscape(null);
        setTopicInsights(null);
        setTimeline(null);
      });

    return () => {
      isMounted = false;
    };
  }, [range, subreddit, topic]);

  useEffect(() => {
    let isMounted = true;
    const params = buildParams();

    fetchJson<SankeyData>(`/api/sankey?${params.toString()}`)
      .then((response) => {
        if (!isMounted) {
          return;
        }
        setSankey(response.data);
      })
      .catch(() => {
        if (!isMounted) {
          return;
        }
        setSankey(null);
      });

    return () => {
      isMounted = false;
    };
  }, [range, subreddit, topic]);

  useEffect(() => {
    const params = new URLSearchParams({ range });
    if (subreddit && subreddit !== "all") {
      params.set("subreddit", subreddit);
    }

    fetchJson<TopicsData>(`/api/topics?${params.toString()}`)
      .then((response) => {
        const list = response.data.topics ?? [];
        setTopicOptions(list);
        setTopic((current) => {
          if (current === "all") {
            return current;
          }
          return list.some((item) => item.topic_id === current) ? current : "all";
        });
      })
      .catch(() => {
        setTopicOptions([]);
      });
  }, [range, subreddit]);

  useEffect(() => {
    fetchJson<SubredditsData>("/api/subreddits?range=all")
      .then((response) => {
        setSubreddits(response.data.subreddits.map((item) => item.subreddit));
      })
      .catch(() => {
        setSubreddits([]);
      });
  }, []);

  const statusLabel = source === "mongo" ? "Live" : "Snapshot";
  const statusDotClass = source === "mongo" ? "status-dot" : "status-dot snapshot";
  const rangeLabel = RANGE_LABELS[range] || range;
  const topicLabel =
    topic === "all"
      ? "All topics"
      : topicOptions.find((item) => item.topic_id === topic)?.topic_name || topic;
  const selectedTopic =
    topic === "all"
      ? null
      : topics?.topics.find((item) => item.topic_id === topic) ??
        topicOptions.find((item) => item.topic_id === topic) ??
        null;
  const topicDescription =
    topic === "all"
      ? ""
      : selectedTopic?.topic_description?.trim() ||
        (loading ? "Loading topic description..." : "No description available.");

  const kpis = useMemo(() => {
    if (!overview) {
      return [];
    }
    return [
      {
        label: "Posts",
        value: formatNumber(overview.posts),
        hint: "Total posts",
      },
      {
        label: "Comments",
        value: formatNumber(overview.comments),
        hint: "Total comments",
      },
      {
        label: "Avg post upvotes",
        value:
          overview.avg_post_upvotes !== null && overview.avg_post_upvotes !== undefined
            ? overview.avg_post_upvotes.toFixed(1)
            : "-",
        hint: "Content approval level",
      },
      {
        label: "Avg comment upvotes/post",
        value:
          overview.avg_comment_upvotes_per_post !== null &&
          overview.avg_comment_upvotes_per_post !== undefined
            ? overview.avg_comment_upvotes_per_post.toFixed(1)
            : "-",
        hint: "Discussion quality",
      },
      {
        label: "Avg comments/post",
        value:
          overview.avg_comments_per_post !== null && overview.avg_comments_per_post !== undefined
            ? overview.avg_comments_per_post.toFixed(1)
            : "-",
        hint: "Interaction volume",
      },
    ];
  }, [overview]);

  return (
    <div className="page">
      <header className="hero reveal reveal-1">
        <div className="hero-tag">
          <span className={statusDotClass} aria-hidden="true" />
          {statusLabel} data
        </div>
        <h1 className="hero-title">Reddit Analyzer</h1>
        <p className="hero-subtitle">A live view of discourse analysis accross Reddit</p>
        <div className="hero-meta">
          <span>Updated: {formatDateTime(updatedAt)}</span>
          <span>Range: {rangeLabel}</span>
          <span>Scope: {subreddit === "all" ? "All subreddits" : subreddit}</span>
          <span>Topic: {topicLabel}</span>
        </div>
      </header>

      <FilterBar
        range={range}
        onRangeChange={setRange}
        subreddit={subreddit}
        onSubredditChange={setSubreddit}
        subreddits={subreddits}
        topic={topic}
        onTopicChange={setTopic}
        topics={topicOptions}
      />

      {error && (
        <div className="card reveal reveal-2">
          <div className="card-title">Data error</div>
          <div className="card-subtitle">{error}</div>
        </div>
      )}

      <section className="kpi-grid reveal reveal-2">
        {kpis.length === 0 && loading ? (
          <>
            {Array.from({ length: 5 }).map((_, idx) => (
              <KpiCard key={idx} label="Loading" value="..." hint="Fetching data" />
            ))}
          </>
        ) : (
          kpis.map((kpi) => (
            <KpiCard key={kpi.label} label={kpi.label} value={kpi.value} hint={kpi.hint} />
          ))
        )}
      </section>

      <section className="grid-full reveal reveal-3">
        <div className="card">
          <div className="card-title">
            Subreddit UMAP
            <span className="pill">UMAP</span>
          </div>
          <div className="card-subtitle">UMAP projection colored by subreddit.</div>
          <div className="chart chart-xl">
            <OpinionMap data={umapLandscapeGlobal} groupBy="subreddit" />
          </div>
        </div>
      </section>

      <section className="grid-full reveal">
        <div className="card">
          <div className="card-title">
            Topic UMAP
            <span className="pill">UMAP</span>
          </div>
          <div className="card-subtitle">UMAP projection colored by topic.</div>
          <div className="chart chart-xl">
            <OpinionMap data={umapLandscapeGlobal} groupBy="topic" gridRight={210} legendRight= {-4}/>
          </div>
        </div>
      </section>

      <section className="grid-two reveal">
        <div className="card">
          <div className="card-title">
            UMAP landscape
            <div className="toggle">
              <button
                className={umapMode === "sentiment" ? "active" : ""}
                onClick={() => setUmapMode("sentiment")}
                type="button"
              >
                Sentiment
              </button>
              <button
                className={umapMode === "stance" ? "active" : ""}
                onClick={() => setUmapMode("stance")}
                type="button"
              >
                Stance
              </button>
            </div>
          </div>
          <div className="card-subtitle">
            Posts colored by {umapMode === "sentiment" ? "sentiment balance" : "stance balance"}.
          </div>
          <div className="chart chart-umap">
            <UmapLandscape data={umapLandscape} mode={umapMode} />
          </div>
        </div>
      </section>

      <section className="grid-two reveal reveal-4">
        <div className="card">
          <div className="card-title">
            Top topics
            <span className="pill">Leaders</span>
          </div>
          <div className="card-subtitle">
            {topic === "all" ? "Most active clusters in the current view" : "Selected topic"}
          </div>
          {topic === "all" ? (
            <TopicTable topics={topics?.topics ?? []} />
          ) : (
            <div className="card-subtitle">{topicDescription}</div>
          )}
        </div>
      </section>

      <section className="section-header reveal">
        <h2>Controversy & Mood</h2>
        <p>How polarization, sentiment, and topic structure move together.</p>
      </section>

      <section className="grid-two reveal">
        <div className="card">
          <div className="card-title">
            Stance x Sentiment
            <span className="pill">Matrix</span>
          </div>
          <div className="card-subtitle">Where agreement and emotion intersect</div>
          <div className="chart">
            <HeatmapChart data={stanceSentiment} />
          </div>
        </div>
        <div className="card">
          <div className="card-title">
            Stance x Sentiment
            <span className="pill">Flow</span>
          </div>
          <div className="card-subtitle">
            Average flow of sentiment into stance for the current filters.
          </div>
          <SankeyPanel data={sankey} />
        </div>
      </section>

      <section className="grid-two reveal">
        <div className="card">
          <div className="card-title">
            Topic stacked bars
            <span className="pill">Stance</span>
          </div>
          <div className="card-subtitle">Sorted by mean polarization score.</div>
          <div className="chart chart-tall">
            <TopicStackedBars data={topicInsights} />
          </div>
        </div>
        <div className="card">
          <div className="card-title">
            Stance trend
            <span className="pill">Timeline</span>
          </div>
          <div className="card-subtitle">Shifts in agreement over time</div>
          <div className="chart">
            <TrendChart data={trends} />
          </div>
        </div>
      </section>

      <section className="grid-two reveal">
        <div className="card">
          <div className="card-title">
            Polarization & mood
            <span className="pill">Weekly</span>
          </div>
          <div className="card-subtitle">
            Weighted by comment weights per snapshot week.
          </div>
          <div className="chart">
            <PolarizationTimeline data={timeline} />
          </div>
        </div>
        <div className="card">
          <div className="card-title">
            Controversy leaderboard
            <div className="toggle">
              <button
                className={insightMode === "post" ? "active" : ""}
                onClick={() => setInsightMode("post")}
                type="button"
              >
                Posts
              </button>
              <button
                className={insightMode === "topic" ? "active" : ""}
                onClick={() => setInsightMode("topic")}
                type="button"
              >
                Topics
              </button>
            </div>
          </div>
          <div className="card-subtitle">Top items by comment weight.</div>
          <div className="chart">
            <ControversyLeaderboard data={leaderboard} />
          </div>
        </div>
      </section>

      <section className="section-header reveal">
        <h2>Showcase Views</h2>
        <p>Alternate lenses for polarization patterns.</p>
      </section>

      <section className="grid-two reveal">
        <div className="card">
          <div className="card-title">
            Opinion fingerprint
            <span className="pill">Radar</span>
          </div>
          <div className="card-subtitle">Ten-axis stance and sentiment profile.</div>
          <OpinionFingerprint data={topicInsights} />
        </div>
      </section>

      <section className="grid-two reveal">
        <div className="card">
          <div className="card-title">
            Polarization vs coherence
            <span className="pill">Topics</span>
          </div>
          <div className="card-subtitle">
            Coherence uses average center distance.
          </div>
          <div className="chart chart-tall">
            <PolarizationCoherence data={topicInsights} />
          </div>
        </div>
      </section>
    </div>
  );
}
