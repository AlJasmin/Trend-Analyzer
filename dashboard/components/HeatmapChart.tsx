"use client";

import ReactECharts from "echarts-for-react";
import type { StanceSentimentData } from "@/lib/types";
import { SENTIMENT_ORDER, STANCE_ORDER } from "@/lib/labels";
import { titleCase } from "@/lib/format";

type HeatmapChartProps = {
  data: StanceSentimentData | null;
};

export default function HeatmapChart({ data }: HeatmapChartProps) {
  const stances = STANCE_ORDER;
  const sentiments = SENTIMENT_ORDER;
  const matrix = data?.matrix ?? [];

  const values = matrix.map((cell) => cell.count);
  const maxValue = values.length ? Math.max(...values) : 0;

  const seriesData = matrix.map((cell) => {
    const x = stances.indexOf(cell.stance);
    const y = sentiments.indexOf(cell.sentiment);
    if (x < 0 || y < 0) {
      return null;
    }
    return [x, y, cell.count];
  }).filter((value): value is number[] => Boolean(value));

  const option = {
    backgroundColor: "transparent",
    grid: { top: 10, right: 12, bottom: 70, left: 68 },
    tooltip: {
      trigger: "item",
      formatter: (params: { data: number[] }) => {
        const [x, y, value] = params.data || [];
        return `${titleCase(sentiments[y] || "")} / ${titleCase(stances[x] || "")}: ${value || 0}`;
      },
    },
    xAxis: {
      type: "category",
      data: stances.map(titleCase),
      axisLabel: { color: "#5e554b" },
      axisLine: { lineStyle: { color: "rgba(27,20,16,0.2)" } },
    },
    yAxis: {
      type: "category",
      data: sentiments.map(titleCase),
      axisLabel: { color: "#5e554b" },
      axisLine: { lineStyle: { color: "rgba(27,20,16,0.2)" } },
    },
    visualMap: {
      
      min: 0,
      max: maxValue || 1,
      orient: "horizontal",
      left: "center",
      bottom: 8,
      text: ["High", "Low"],
      textStyle: { color: "#5e554b" },
      inRange: {
        color: ["#f6f1e8", "#f4d35e", "#e4572e"],
      },
    },
    series: [
      {
        type: "heatmap",
        data: seriesData,
        label: { show: false },
        emphasis: {
          itemStyle: {
            shadowBlur: 12,
            shadowColor: "rgba(0,0,0,0.25)",
          },
        },
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />;
}
