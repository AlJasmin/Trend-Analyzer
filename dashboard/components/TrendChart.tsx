"use client";

import ReactECharts from "echarts-for-react";
import type { TrendData } from "@/lib/types";
import { STANCE_COLORS, STANCE_ORDER } from "@/lib/labels";
import { titleCase } from "@/lib/format";

type TrendChartProps = {
  data: TrendData | null;
};

export default function TrendChart({ data }: TrendChartProps) {
  const points = data?.stance_series ?? [];
  const dates = points.map((point) => point.date);
  const keySet = new Set<string>();
  points.forEach((point) => {
    Object.keys(point.counts).forEach((key) => keySet.add(key));
  });

  const keys = [
    ...STANCE_ORDER.filter((key) => keySet.has(key)),
    ...Array.from(keySet).filter((key) => !STANCE_ORDER.includes(key)),
  ];

  const series = keys.map((key) => ({
    name: titleCase(key),
    type: "line",
    stack: "total",
    areaStyle: { opacity: 0.35 },
    emphasis: { focus: "series" },
    lineStyle: { width: 2 },
    symbol: "none",
    data: points.map((point) => point.counts[key] || 0),
    color: STANCE_COLORS[key] || "#a68a64",
  }));

  const option = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    legend: { bottom: 0, textStyle: { color: "#5e554b" } },
    grid: { top: 20, right: 20, bottom: 40, left: 40 },
    xAxis: {
      type: "category",
      data: dates,
      axisLabel: {
        color: "#5e554b",
        formatter: (value: string) => {
          const date = new Date(value);
          return Number.isNaN(date.getTime())
            ? value
            : date.toLocaleDateString("en-US", { month: "short", day: "2-digit" });
        },
      },
      axisLine: { lineStyle: { color: "rgba(27,20,16,0.2)" } },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#5e554b" },
      splitLine: { lineStyle: { color: "rgba(27,20,16,0.08)" } },
    },
    series,
  };

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />;
}
