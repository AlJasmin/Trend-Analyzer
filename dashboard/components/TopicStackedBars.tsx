"use client";

import ReactECharts from "echarts-for-react";
import type { TopicInsightsData } from "@/lib/types";
import { STANCE_COLORS } from "@/lib/labels";
import { titleCase } from "@/lib/format";

type TopicStackedBarsProps = {
  data: TopicInsightsData | null;
};

const STANCE_KEYS = ["agree", "neutral", "disagree"];

export default function TopicStackedBars({ data }: TopicStackedBarsProps) {
  if (!data || data.topics.length === 0) {
    return <div className="card-subtitle">No topic distribution data.</div>;
  }

  const topics = data.topics.slice(0, 12);
  const labels = topics.map((topic) => topic.topic_name || topic.topic_id);

  const series = STANCE_KEYS.map((key) => ({
    name: titleCase(key),
    type: "bar",
    stack: "total",
    emphasis: { focus: "series" },
    itemStyle: { color: STANCE_COLORS[key] || "#a68a64" },
    data: topics.map((topic) => topic.stance_dist[key] || 0),
  }));

  const option = {
    backgroundColor: "transparent",
    grid: { top: 10, right: 20, bottom: 20, left: 120 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLabel: {
        color: "#5e554b",
        formatter: (value: number) => `${Math.round(value * 100)}%`,
      },
      splitLine: { lineStyle: { color: "rgba(27,20,16,0.08)" } },
    },
    yAxis: {
      type: "category",
      data: labels,
      axisLabel: { color: "#5e554b", width: 110, overflow: "truncate" },
      axisLine: { lineStyle: { color: "rgba(27,20,16,0.2)" } },
    },
    series,
  };

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />;
}
