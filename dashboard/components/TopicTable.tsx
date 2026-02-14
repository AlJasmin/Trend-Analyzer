import type { TopicItem } from "@/lib/types";
import { formatNumber } from "@/lib/format";

type TopicTableProps = {
  topics: TopicItem[];
};

function trimText(text: string, limit: number) {
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, limit)}...`;
}

export default function TopicTable({ topics }: TopicTableProps) {
  if (!topics.length) {
    return <div className="card-subtitle">No topics found.</div>;
  }

  const top = [...topics].sort((a, b) => b.count - a.count).slice(0, 12);

  return (
    <div className="topic-table">
      {top.map((topic) => (
        <div className="topic-row" key={topic.topic_id}>
          <strong>{topic.topic_name || topic.topic_id}</strong>
          <div className="topic-meta">
            <span>{formatNumber(topic.count)} posts</span>
            <span>{formatNumber(topic.avg_comments)} avg comments</span>
          </div>
          {topic.topic_description ? (
            <div className="card-subtitle" title={topic.topic_description}>
              {trimText(topic.topic_description, 120)}
            </div>
          ) : null}
        </div>
      ))}
    </div>
  );
}
