import type { OverviewData } from "@/lib/types";
import { formatNumber, formatPercent } from "@/lib/format";

type QualityPanelProps = {
  overview: OverviewData | null;
};

export default function QualityPanel({ overview }: QualityPanelProps) {
  if (!overview) {
    return <div className="card-subtitle">Waiting for metrics.</div>;
  }

  const total = overview.comments || 0;
  const unlabeled = Math.max(total - overview.labeled_comments, 0);
  const deleted = overview.deleted_comments || 0;
  const lowConfidence = overview.low_confidence_comments || 0;

  const rows = [
    {
      label: "Unlabeled",
      count: unlabeled,
    },
    {
      label: "Deleted or removed",
      count: deleted,
    },
    {
      label: "Low confidence",
      count: lowConfidence,
    },
  ];

  return (
    <div className="quality-bars">
      {rows.map((row) => {
        const ratio = total ? row.count / total : 0;
        return (
          <div className="quality-bar" key={row.label}>
            <span>
              {row.label}: {formatNumber(row.count)} ({formatPercent(ratio)})
            </span>
            <div className="quality-track">
              <div className="quality-fill" style={{ width: `${ratio * 100}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
