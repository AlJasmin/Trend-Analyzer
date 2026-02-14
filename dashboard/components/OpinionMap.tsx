"use client";

import ReactECharts from "echarts-for-react";
import type { UmapData } from "@/lib/types";
import { formatNumber } from "@/lib/format";

type OpinionMapProps = {
  data: UmapData | null;
  groupBy?: "subreddit" | "topic";
  legendRight?: number;
  gridRight?: number;
};

const COLOR_POOL = [
  "#2d9c8b",
  "#e4572e",
  "#f4d35e",
  "#6d6875",
  "#3b82f6",
  "#22c55e",
  "#f97316",
  "#6366f1",
  "#ec4899",
  "#14b8a6",
  "#84cc16",
  "#ef4444",
  "#0ea5e9",
  "#a855f7",
  "#64748b",
  "#f59e0b",
];

function normalizeLabel(value: string | null | undefined) {
  if (!value) {
    return "unknown";
  }
  const trimmed = value.trim();
  return trimmed.length ? trimmed : "unknown";
}

export default function OpinionMap({
  data,
  groupBy = "subreddit",
  legendRight,
  gridRight,
}: OpinionMapProps) {
  if (!data || data.points.length === 0) {
    return <div className="card-subtitle">No UMAP points available.</div>;
  }

  const groups = new Map<
    string,
    { label: string; points: { name: string; value: number[]; size: number }[] }
  >();

  data.points.forEach((point) => {
    const key =
      groupBy === "topic"
        ? normalizeLabel(point.topic_name || point.topic_id)
        : normalizeLabel(point.subreddit);
    const label = key === "unknown" ? "Unknown" : groupBy === "topic" ? key : `r/${key}`;
    if (!groups.has(key)) {
      groups.set(key, { label, points: [] });
    }
    groups.get(key)!.points.push({
      name: point.label || point.id,
      value: [point.x, point.y],
      size: point.size || 0,
    });
  });

  const groupList = Array.from(groups.values()).sort(
    (a, b) => b.points.length - a.points.length,
  );

  const series = groupList.map((group, index) => ({
    name: group.label,
    type: "scatter",
    data: group.points,
    symbolSize: 6,
    itemStyle: { color: COLOR_POOL[index % COLOR_POOL.length], opacity: 0.85 },
  }));

  const legendRightValue = legendRight ?? 0;
  const gridRightValue = gridRight ?? 180;

  const option = {
    backgroundColor: "transparent",
    grid: { top: 10, right: gridRightValue, bottom: 10, left: 10 },
    tooltip: {
      formatter: (params: { name: string; seriesName: string; data?: { size?: number } }) => {
        const size = params.data?.size ?? 0;
        const groupLabel = groupBy === "topic" ? "Topic" : "Subreddit";
        return `${params.name}<br/>${groupLabel}: ${params.seriesName}<br/>Comments: ${formatNumber(
          size,
        )}`;
      },
    },
    legend: {
      type: "scroll",
      orient: "vertical",
      right: legendRightValue,
      top: 10,
      bottom: 10,
      textStyle: { color: "#5e554b", fontSize: 11 },
      pageTextStyle: { color: "#5e554b" },
      pageIconColor: "#5e554b",
      pageIconInactiveColor: "rgba(27,20,16,0.3)",
    },
    xAxis: {
      type: "value",
      axisLabel: { show: false },
      axisLine: { show: false },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      axisLabel: { show: false },
      axisLine: { show: false },
      splitLine: { show: false },
    },
    series,
  };

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />;
}
