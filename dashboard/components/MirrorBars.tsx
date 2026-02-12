"use client";

import ReactECharts from "echarts-for-react";
import type { TopicInsightsData } from "@/lib/types";
import { SENTIMENT_COLORS, STANCE_COLORS } from "@/lib/labels";

type MirrorBarsProps = {
  data: TopicInsightsData | null;
};

export default function MirrorBars({ data }: MirrorBarsProps) {
  if (!data || data.topics.length === 0) {
    return <div className="card-subtitle">No mirror bar data.</div>;
  }

  const topics = data.topics.slice(0, 10);
  const labels = topics.map((topic) => topic.topic_name || topic.topic_id);

  const sentimentPos = topics.map((topic) => -(topic.sentiment_dist.positive || 0));
  const sentimentNeg = topics.map((topic) => -(topic.sentiment_dist.negative || 0));
  const stanceAgree = topics.map((topic) => topic.stance_dist.agree || 0);
  const stanceDisagree = topics.map((topic) => topic.stance_dist.disagree || 0);

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: { seriesName: string; value: number }[]) =>
        params
          .map((item) => `${item.seriesName}: ${Math.abs(item.value * 100).toFixed(1)}%`)
          .join("<br/>"),
    },
    grid: { top: 10, right: 20, bottom: 30, left: 120 },
    xAxis: {
      type: "value",
      min: -1,
      max: 1,
      axisLabel: {
        color: "#5e554b",
        formatter: (value: number) => `${Math.abs(value * 100)}%`,
      },
      splitLine: { lineStyle: { color: "rgba(27,20,16,0.08)" } },
    },
    yAxis: {
      type: "category",
      data: labels,
      axisLabel: { color: "#5e554b", width: 110, overflow: "truncate" },
      axisLine: { lineStyle: { color: "rgba(27,20,16,0.2)" } },
    },
    series: [
      {
        name: "Sentiment +",
        type: "bar",
        stack: "sentiment",
        data: sentimentPos,
        itemStyle: { color: SENTIMENT_COLORS.positive || "#2d9c8b" },
      },
      {
        name: "Sentiment -",
        type: "bar",
        stack: "sentiment",
        data: sentimentNeg,
        itemStyle: { color: SENTIMENT_COLORS.negative || "#e4572e" },
      },
      {
        name: "Stance Agree",
        type: "bar",
        stack: "stance",
        data: stanceAgree,
        itemStyle: { color: STANCE_COLORS.agree || "#2d9c8b" },
      },
      {
        name: "Stance Disagree",
        type: "bar",
        stack: "stance",
        data: stanceDisagree,
        itemStyle: { color: STANCE_COLORS.disagree || "#e4572e" },
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />;
}
