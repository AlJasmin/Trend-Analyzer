"use client";

import ReactECharts from "echarts-for-react";
import type { SankeyData } from "@/lib/types";
import { titleCase } from "@/lib/format";

type SankeyPanelProps = {
  data: SankeyData | null;
};

const SENTIMENT_PREFIX = "sentiment:";
const STANCE_PREFIX = "stance:";

function formatNodeLabel(value: string) {
  return titleCase(value.replace(/^sentiment:|^stance:/, ""));
}

function withPrefix(value: string, prefix: string) {
  if (value.startsWith(prefix)) {
    return value;
  }
  return `${prefix}${value}`;
}

export default function SankeyPanel({ data }: SankeyPanelProps) {
  if (!data || data.facets.length === 0) {
    return <div className="card-subtitle">No flow data available.</div>;
  }

  return (
    <div className="sankey-grid">
      {data.facets.map((facet) => {
        const nodeSet = new Set<string>();
        const links = facet.links.map((link) => {
          const source = withPrefix(link.source, SENTIMENT_PREFIX);
          const target = withPrefix(link.target, STANCE_PREFIX);
          nodeSet.add(source);
          nodeSet.add(target);
          return { source, target, value: link.value };
        });
        const nodes = Array.from(nodeSet).map((name) => ({ name }));

        const option = {
          backgroundColor: "transparent",
          tooltip: {
            formatter: (params: {
              name?: string;
              data?: { source?: string; target?: string; value?: number };
            }) => {
              if (!params.data) {
                return params.name ? formatNodeLabel(params.name) : "";
              }
              const { source, target, value } = params.data;
              if (!source || !target) {
                return "";
              }
              return `${formatNodeLabel(source)} \u2192 ${formatNodeLabel(target)}: ${Math.round(
                value || 0,
              )}`;
            },
          },
          series: [
            {
              type: "sankey",
              top: 8,
              bottom: 8,
              left: 8,
              right: 8,
              data: nodes,
              links,
              lineStyle: {
                color: "source",
                curveness: 0.5,
                opacity: 0.6,
              },
              label: {
                color: "#1b1410",
                fontSize: 11,
                formatter: (params: { name: string }) => formatNodeLabel(params.name),
              },
            },
          ],
        };

        return (
          <div className="sankey-card" key={facet.key}>
            <div className="sankey-title">{facet.label}</div>
            <ReactECharts option={option} style={{ height: "340px", width: "100%" }} />
          </div>
        );
      })}
    </div>
  );
}
