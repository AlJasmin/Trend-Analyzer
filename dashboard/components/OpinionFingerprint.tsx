"use client";

import ReactECharts from "echarts-for-react";
import type { TopicInsightsData } from "@/lib/types";

type OpinionFingerprintProps = {
  data: TopicInsightsData | null;
};

const INDICATORS = [
  { name: "Agree", max: 1 },
  { name: "Disagree", max: 1 },
  { name: "Neutral (St)", max: 1 },
  { name: "Partial", max: 1 },
  { name: "Unknown (St)", max: 1 },
  { name: "Positive", max: 1 },
  { name: "Negative", max: 1 },
  { name: "Neutral (Se)", max: 1 },
  { name: "Mixed", max: 1 },
  { name: "Unknown (Se)", max: 1 },
];

export default function OpinionFingerprint({ data }: OpinionFingerprintProps) {
  if (!data || data.topics.length === 0) {
    return <div className="card-subtitle">No fingerprint data available.</div>;
  }

  const topics = data.topics.slice(0, 10);

  return (
    <div className="fingerprint-grid">
      {topics.map((topic) => {
        const values = [
          topic.stance_dist.agree || 0,
          topic.stance_dist.disagree || 0,
          topic.stance_dist.neutral || 0,
          topic.stance_dist.partial || 0,
          topic.stance_dist.unknown || 0,
          topic.sentiment_dist.positive || 0,
          topic.sentiment_dist.negative || 0,
          topic.sentiment_dist.neutral || 0,
          topic.sentiment_dist.mixed || 0,
          topic.sentiment_dist.unknown || 0,
        ];

        const option = {
          backgroundColor: "transparent",
          radar: {
            indicator: INDICATORS,
            splitNumber: 4,
            axisName: { color: "#5e554b", fontSize: 9 },
            splitLine: { lineStyle: { color: "rgba(27,20,16,0.1)" } },
            splitArea: { areaStyle: { color: ["rgba(255,255,255,0.2)"] } },
            axisLine: { lineStyle: { color: "rgba(27,20,16,0.2)" } },
          },
          series: [
            {
              type: "radar",
              data: [
                {
                  value: values,
                  areaStyle: { color: "rgba(45,156,139,0.2)" },
                  lineStyle: { color: "#2d9c8b", width: 2 },
                  symbol: "none",
                },
              ],
            },
          ],
        };

        return (
          <div className="mini-card" key={topic.topic_id}>
            <div className="mini-title">{topic.topic_name || topic.topic_id}</div>
            <div className="fingerprint-chart">
              <ReactECharts option={option} style={{ height: "180px", width: "100%" }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
