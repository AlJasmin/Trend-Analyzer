"use client";

import { useMemo, useState } from "react";
import type { TopicItem } from "@/lib/types";
import { hashString } from "@/lib/hash";
import { formatNumber } from "@/lib/format";

const COLORS = [
  "#e4572e",
  "#2d9c8b",
  "#f4d35e",
  "#6d6875",
  "#ed6a5a",
  "#00a6a6",
  "#9a8c98",
  "#f2a65a",
];

type TopicMapProps = {
  topics: TopicItem[];
};

type Node = {
  x: number;
  y: number;
  size: number;
  color: string;
  label: string;
  detail: TopicItem;
};

export default function TopicMap({ topics }: TopicMapProps) {
  const [hovered, setHovered] = useState<TopicItem | null>(null);

  const nodes = useMemo<Node[]>(() => {
    if (!topics.length) {
      return [];
    }
    const list = [...topics].sort((a, b) => b.count - a.count).slice(0, 42);
    const maxCount = Math.max(...list.map((item) => item.count));
    const maxDistance = Math.max(
      ...list.map((item) => item.avg_center_distance || 0),
      1,
    );

    return list.map((item, index) => {
      const seed = hashString(item.topic_id || item.topic_name || String(index));
      const angle = ((seed % 360) / 360) * Math.PI * 2;
      const distance = item.avg_center_distance ?? index / list.length;
      const radius = 40 + (distance / maxDistance) * 140;
      const jitter = ((seed >> 4) % 100) / 100 - 0.5;
      const x = 240 + Math.cos(angle) * (radius + jitter * 18);
      const y = 190 + Math.sin(angle) * (radius + jitter * 18);
      const size = 6 + Math.sqrt(item.count / maxCount) * 22;
      const color = COLORS[seed % COLORS.length];
      return {
        x,
        y,
        size,
        color,
        label: item.topic_name || item.topic_id,
        detail: item,
      };
    });
  }, [topics]);

  if (!topics.length) {
    return <div className="card-subtitle">No topic data available.</div>;
  }

  return (
    <div className="topic-map-wrapper">
      <svg viewBox="0 0 480 380" width="100%" height="100%">
        <defs>
          <radialGradient id="glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(244,211,94,0.35)" />
            <stop offset="100%" stopColor="rgba(244,211,94,0)" />
          </radialGradient>
        </defs>
        <circle cx="240" cy="190" r="180" fill="url(#glow)" />
        <circle cx="240" cy="190" r="120" fill="none" stroke="rgba(27,20,16,0.08)" />
        <circle cx="240" cy="190" r="60" fill="none" stroke="rgba(27,20,16,0.08)" />
        {nodes.map((node) => (
          <g
            key={node.label}
            onMouseEnter={() => setHovered(node.detail)}
            onMouseLeave={() => setHovered(null)}
          >
            <circle
              cx={node.x}
              cy={node.y}
              r={node.size}
              fill={node.color}
              fillOpacity={0.75}
              stroke="rgba(27,20,16,0.2)"
            >
              <title>
                {node.label} ({formatNumber(node.detail.count)} posts)
              </title>
            </circle>
          </g>
        ))}
      </svg>
      <div className="topic-detail">
        {hovered ? (
          <>
            <strong>{hovered.topic_name || hovered.topic_id}</strong>
            <span>{formatNumber(hovered.count)} posts</span>
            {hovered.topic_description ? (
              <p>{hovered.topic_description}</p>
            ) : (
              <p>No description available.</p>
            )}
          </>
        ) : (
          <>
            <strong>Explore topics</strong>
            <p>Hover a node to preview topic details.</p>
          </>
        )}
      </div>
    </div>
  );
}
