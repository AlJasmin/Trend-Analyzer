import { getDb } from "@/lib/mongo";
import { buildCommentPipeline, buildPostMatch, parseFilters } from "@/lib/filters";
import { respondWithFallback } from "@/lib/response";

export async function GET(req: Request) {
  const filters = parseFilters(req);

  return respondWithFallback(
    "overview",
    { range: filters.range, subreddit: filters.subreddit, topic: filters.topic },
    async () => {
      const db = await getDb();
      const postsMatch = buildPostMatch(filters);
      const postPipeline: Record<string, unknown>[] = [];
      if (Object.keys(postsMatch).length > 0) {
        postPipeline.push({ $match: postsMatch });
      }
      postPipeline.push({
        $group: {
          _id: null,
          posts: { $sum: 1 },
          avg_comments: { $avg: "$num_comments" },
          avg_post_upvotes: { $avg: "$score" },
        },
      });

      const commentPipeline = buildCommentPipeline(filters);
      const normalizedText = {
        $toLower: { $trim: { input: { $ifNull: ["$comment_text", ""] } } },
      };
      const hasStance = {
        $and: [{ $ne: ["$stance_label", null] }, { $ne: ["$stance_label", ""] }],
      };
      const hasSentiment = {
        $and: [{ $ne: ["$sentiment_label", null] }, { $ne: ["$sentiment_label", ""] }],
      };
      const isLabeled = { $and: [hasStance, hasSentiment] };
      const isDeleted = {
        $in: [normalizedText, ["[deleted]", "[removed]", ""]],
      };

      commentPipeline.push({
        $project: {
          comment_text: 1,
          stance_label: 1,
          sentiment_label: 1,
          llm_confidence: 1,
        },
      });
      commentPipeline.push({
        $group: {
          _id: null,
          comments: { $sum: 1 },
          labeled: { $sum: { $cond: [isLabeled, 1, 0] } },
          deleted: { $sum: { $cond: [isDeleted, 1, 0] } },
          low_confidence: {
            $sum: {
              $cond: [
                { $and: [isLabeled, { $lt: ["$llm_confidence", 0.5] }] },
                1,
                0,
              ],
            },
          },
          confidence_sum: {
            $sum: {
              $cond: [
                { $and: [isLabeled, { $ne: ["$llm_confidence", null] }] },
                "$llm_confidence",
                0,
              ],
            },
          },
          confidence_count: {
            $sum: {
              $cond: [
                { $and: [isLabeled, { $ne: ["$llm_confidence", null] }] },
                1,
                0,
              ],
            },
          },
        },
      });

      const [postStats] = await db.collection("posts").aggregate(postPipeline).toArray();
      const [commentStats] = await db.collection("comments").aggregate(commentPipeline).toArray();
      const commentUpvotesPipeline = buildCommentPipeline(filters);
      commentUpvotesPipeline.push({
        $project: {
          post_id: 1,
          score: { $ifNull: ["$score", { $ifNull: ["$upvote_score", 0] }] },
        },
      });
      commentUpvotesPipeline.push({
        $group: {
          _id: "$post_id",
          comment_upvotes_sum: { $sum: "$score" },
        },
      });
      commentUpvotesPipeline.push({
        $group: {
          _id: null,
          avg_comment_upvotes_per_post: { $avg: "$comment_upvotes_sum" },
        },
      });
      const [commentUpvotes] = await db
        .collection("comments")
        .aggregate(commentUpvotesPipeline)
        .toArray();

      const posts = postStats?.posts || 0;
      const comments = commentStats?.comments || 0;
      const labeled_comments = commentStats?.labeled || 0;
      const label_coverage = comments ? labeled_comments / comments : 0;
      const avg_confidence =
        commentStats?.confidence_count > 0
          ? commentStats.confidence_sum / commentStats.confidence_count
          : null;

      return {
        posts,
        comments,
        labeled_comments,
        label_coverage,
        avg_confidence,
        avg_comments_per_post: postStats?.avg_comments ?? null,
        avg_post_upvotes: postStats?.avg_post_upvotes ?? null,
        avg_comment_upvotes_per_post: commentUpvotes?.avg_comment_upvotes_per_post ?? null,
        deleted_comments: commentStats?.deleted || 0,
        low_confidence_comments: commentStats?.low_confidence || 0,
      };
    },
  );
}
