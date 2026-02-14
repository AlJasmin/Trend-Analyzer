import type { TopicItem } from "@/lib/types";

type FilterBarProps = {
  range: string;
  onRangeChange: (value: string) => void;
  subreddit: string;
  onSubredditChange: (value: string) => void;
  subreddits: string[];
  topic: string;
  onTopicChange: (value: string) => void;
  topics: TopicItem[];
};

const RANGE_OPTIONS = [
  { value: "7d", label: "Last 7 days" },
  { value: "30d", label: "Last 30 days" },
  { value: "90d", label: "Last 90 days" },
  { value: "180d", label: "Last 180 days" },
  { value: "365d", label: "Last 12 months" },
  { value: "all", label: "All time" },
];

export default function FilterBar({
  range,
  onRangeChange,
  subreddit,
  onSubredditChange,
  subreddits,
  topic,
  onTopicChange,
  topics,
}: FilterBarProps) {
  return (
    <div className="filters reveal reveal-2">
      <div className="filter">
        <label htmlFor="range">Time range</label>
        <select
          id="range"
          value={range}
          onChange={(event) => onRangeChange(event.target.value)}
        >
          {RANGE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
      <div className="filter">
        <label htmlFor="subreddit">Subreddit</label>
        <select
          id="subreddit"
          value={subreddit}
          onChange={(event) => onSubredditChange(event.target.value)}
        >
          <option value="all">All subreddits</option>
          {subreddits.map((name) => (
            <option key={name} value={name}>
              r/{name}
            </option>
          ))}
        </select>
      </div>
      <div className="filter">
        <label htmlFor="topic">Topic</label>
        <select id="topic" value={topic} onChange={(event) => onTopicChange(event.target.value)}>
          <option value="all">All topics</option>
          {topics.map((item) => (
            <option key={item.topic_id} value={item.topic_id}>
              {item.topic_name || item.topic_id}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
