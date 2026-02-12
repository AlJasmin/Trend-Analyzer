"use client";

import ReactECharts from "echarts-for-react";
import type { LeaderboardData } from "@/lib/types";
import { formatPercent } from "@/lib/format";

type ControversyLeaderboardProps = {
  data: LeaderboardData | null;
};

export default function ControversyLeaderboard({ data }: ControversyLeaderboardProps) {
  if (!data || data.items.length === 0) {
    return <div className="card-subtitle">No leaderboard data available.</div>;
  }

  const maxSize = Math.max(...data.items.map((item) => item.comment_weight_sum), 1);
  const seriesData = data.items.map((item) => ({
    name: item.label,
    value: [item.polarization_score, item.comment_weight_sum],
  }));

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      formatter: (params: { name: string; value: number[] }) => {
        const [polarization, weight] = params.value;
        return `${params.name}<br/>Polarization: ${formatPercent(
          polarization,
        )}<br/>Weight sum: ${Math.round(weight)}`;
      },
    },
    grid: { top: 20, right: 20, bottom: 40, left: 50 },
    xAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLabel: { color: "#5e554b", formatter: (value: number) => `${value.toFixed(2)}` },
      axisLine: { lineStyle: { color: "rgba(27,20,16,0.2)" } },
      splitLine: { lineStyle: { color: "rgba(27,20,16,0.08)" } },
      name: "Polarization",
      nameLocation: "middle",
      nameGap: 28,
      nameTextStyle: { color: "#5e554b", fontSize: 12 },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#5e554b" },
      axisLine: { lineStyle: { color: "rgba(27,20,16,0.2)" } },
      splitLine: { lineStyle: { color: "rgba(27,20,16,0.08)" } },
      name: "Comment weight",
      nameLocation: "middle",
      nameGap: 40,
      nameTextStyle: { color: "#5e554b", fontSize: 12 },
    },
    series: [
      {
        type: "scatter",
        data: seriesData,
        symbolSize: (value: number[]) => {
          const weight = value[1] || 0;
          return 6 + Math.sqrt(weight / maxSize) * 24;
        },
        label: { show: false },
        emphasis: {
          label: {
            show: true,
            formatter: (params: { name: string }) => params.name,
            color: "#1b1410",
            fontSize: 11,
          },
        },
        itemStyle: { color: "#2d9c8b", opacity: 0.8 },
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />;
}
