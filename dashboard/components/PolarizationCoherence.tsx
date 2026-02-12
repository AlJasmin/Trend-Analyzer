"use client";

import ReactECharts from "echarts-for-react";
import type { TopicInsightsData } from "@/lib/types";
import { formatPercent } from "@/lib/format";

type PolarizationCoherenceProps = {
  data: TopicInsightsData | null;
};

export default function PolarizationCoherence({ data }: PolarizationCoherenceProps) {
  if (!data || data.topics.length === 0) {
    return <div className="card-subtitle">No coherence data.</div>;
  }

  const points = data.topics
    .filter((topic) => topic.avg_center_distance !== null)
    .map((topic) => ({
      name: topic.topic_name || topic.topic_id,
      value: [topic.polarization_score, topic.avg_center_distance],
    }));

  if (points.length === 0) {
    return <div className="card-subtitle">No coherence distances available.</div>;
  }

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      formatter: (params: { name: string; value: number[] }) => {
        const [polarization, distance] = params.value;
        return `${params.name}<br/>Polarization: ${formatPercent(
          polarization,
        )}<br/>Center distance: ${distance.toFixed(3)}`;
      },
    },
    grid: { top: 20, right: 20, bottom: 40, left: 50 },
    xAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLabel: { color: "#5e554b" },
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
      name: "Coherence",
      nameLocation: "middle",
      nameGap: 40,
      nameTextStyle: { color: "#5e554b", fontSize: 12 },
    },
    series: [
      {
        type: "scatter",
        data: points,
        symbolSize: 10,
        itemStyle: { color: "#f4d35e", opacity: 0.85 },
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />;
}
