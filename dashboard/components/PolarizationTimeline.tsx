"use client";

import ReactECharts from "echarts-for-react";
import type { PolarizationTimelineData } from "@/lib/types";
import { formatPercent } from "@/lib/format";

type PolarizationTimelineProps = {
  data: PolarizationTimelineData | null;
};

export default function PolarizationTimeline({ data }: PolarizationTimelineProps) {
  if (!data || data.points.length === 0) {
    return <div className="card-subtitle">No timeline data available.</div>;
  }

  const weeks = data.points.map((point) => point.week);
  const polarization = data.points.map((point) => point.polarization_score);
  const sentiment = data.points.map((point) => point.sentiment_balance);

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      formatter: (params: { seriesName: string; value: number }[]) => {
        const lines = params.map(
          (item) => `${item.seriesName}: ${formatPercent(item.value)}`,
        );
        return lines.join("<br/>");
      },
    },
    legend: { bottom: 0, textStyle: { color: "#5e554b" } },
    grid: { top: 20, right: 40, bottom: 40, left: 40 },
    xAxis: {
      type: "category",
      data: weeks,
      axisLabel: { color: "#5e554b" },
      axisLine: { lineStyle: { color: "rgba(27,20,16,0.2)" } },
    },
    yAxis: [
      {
        type: "value",
        min: 0,
        max: 1,
        axisLabel: {
          color: "#5e554b",
          formatter: (value: number) => `${Math.round(value * 100)}%`,
        },
        splitLine: { lineStyle: { color: "rgba(27,20,16,0.08)" } },
      },
      {
        type: "value",
        min: -1,
        max: 1,
        axisLabel: {
          color: "#5e554b",
          formatter: (value: number) => `${Math.round(value * 100)}%`,
        },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: "Polarization",
        type: "line",
        yAxisIndex: 0,
        data: polarization,
        smooth: true,
        lineStyle: { color: "#e4572e", width: 2 },
        symbol: "none",
        areaStyle: { opacity: 0.12 },
      },
      {
        name: "Sentiment balance",
        type: "line",
        yAxisIndex: 1,
        data: sentiment,
        smooth: true,
        lineStyle: { color: "#2d9c8b", width: 2 },
        symbol: "none",
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />;
}
