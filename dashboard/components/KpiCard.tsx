type KpiCardProps = {
  label: string;
  value: string;
  hint?: string;
};

export default function KpiCard({ label, value, hint }: KpiCardProps) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {hint ? <div className="kpi-hint">{hint}</div> : null}
    </div>
  );
}
