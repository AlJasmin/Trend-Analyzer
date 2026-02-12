"use client";

import ReactECharts from "echarts-for-react";
import type { UmapData } from "@/lib/types";
import { formatPercent } from "@/lib/format";

type UmapLandscapeProps = {
  data: UmapData | null;
  mode: "sentiment" | "stance";
};

export default function UmapLandscape({ data, mode }: UmapLandscapeProps) {
  if (!data || data.points.length === 0) {
    return <div className="card-subtitle">No UMAP points available.</div>;
  }

  const maxSize = Math.max(...data.points.map((point) => point.size || 0), 1);
  const seriesData = data.points.map((point) => ({
    name: point.label,
    value: [
      point.x,
      point.y,
      mode === "sentiment" ? point.sentiment_balance ?? 0 : point.stance_balance ?? 0,
      point.size,
    ],
  }));
  const metricLabel = mode === "sentiment" ? "Sentiment balance" : "Stance balance";
  const legendMaxLabel = mode === "sentiment" ? "Positive" : "Agree";
  const legendMinLabel = mode === "sentiment" ? "Negative" : "Disagree";

  const option = {
    backgroundColor: "transparent",
    grid: { top: 10, right: 10, bottom: 40, left: 10 },
    tooltip: {
      formatter: (params: { name: string; value: number[] }) => {
        const [, , metricValue, size] = params.value;
        return `${params.name}<br/>${metricLabel}: ${formatPercent(
          metricValue,
        )}<br/>Comments: ${Math.round(
          size,
        )}`;
      },
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
    visualMap: {
      dimension: 2,
      min: -1,
      max: 1,
      show: true,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      text: [legendMaxLabel, legendMinLabel],
      textStyle: { color: "#5e554b" },
      inRange: {
        color: ["#e4572e", "#b9b1a6", "#2d9c8b"],
      },
    },
    series: [
      {
        type: "scatter",
        data: seriesData,
        symbolSize: (value: number[]) => {
          const size = value[3] || 0;
          return 5 + Math.sqrt(size / maxSize) * 22;
        },
        itemStyle: { opacity: 0.85 },
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />;
}
