import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

schema_sql = """
CREATE TABLE IF NOT EXISTS dim_author (
    authorId TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS dim_channel (
    channelId TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS dim_date (
    dateId TEXT PRIMARY KEY,
    date DATE,
    year INT,
    month INT,
    day INT,
    weekday INT,
    monthName TEXT,
    quarter INT
);

CREATE TABLE IF NOT EXISTS dim_hashtag (
    hashtagId TEXT PRIMARY KEY,
    hashtag TEXT
);

CREATE TABLE IF NOT EXISTS dim_topiccategory (
    topicCategoryId TEXT PRIMARY KEY,
    topicCategory TEXT
);

CREATE TABLE IF NOT EXISTS dim_video (
    videoId TEXT PRIMARY KEY,
    channelId TEXT REFERENCES dim_channel(channelId),
    dateId TEXT REFERENCES dim_date(dateId),
    viewCount INT,
    likeCount_video INT,
    commentCount INT,
    engagement_rate FLOAT,
    contentDuration INT
);

CREATE TABLE IF NOT EXISTS fact_comments (
    commentId TEXT PRIMARY KEY,
    videoId TEXT REFERENCES dim_video(videoId),
    authorId TEXT REFERENCES dim_author(authorId),
    channelId TEXT REFERENCES dim_channel(channelId),
    dateId TEXT REFERENCES dim_date(dateId),
    likeCount_comment INT,
    is_spam BOOLEAN,
    spam_score FLOAT,
    weighted_relevance FLOAT,
    is_relevant BOOLEAN,
    cluster_label TEXT,
    sentiment_label TEXT,
    sentiment_score FLOAT,
    actionability_label TEXT,
    actionability_score FLOAT,
    product_resonance_score FLOAT
);

CREATE TABLE IF NOT EXISTS bridge_hashtag (
    hashtagId TEXT REFERENCES dim_hashtag(hashtagId),
    videoId TEXT REFERENCES dim_video(videoId),
    commentId TEXT REFERENCES fact_comments(commentId),
    PRIMARY KEY (hashtagId, videoId, commentId)
);

CREATE TABLE IF NOT EXISTS bridge_topic_category (
    topicCategoryId TEXT REFERENCES dim_topiccategory(topicCategoryId),
    videoId TEXT REFERENCES dim_video(videoId),
    PRIMARY KEY (topicCategoryId, videoId)
);
"""

def get_total_video_analyzed():
    """Return total number of videos in dim_video"""
    result = supabase.table("dim_video").select("videoid", count="exact").execute()
    return result.count or 0

def get_average_engagement_rate():
    """Return average engagement_rate from dim_video"""
    result = supabase.table("dim_video").select("engagement_rate").execute()
    if not result.data:
        return 0
    return sum(r["engagement_rate"] for r in result.data) / len(result.data)

def get_average_prs_score():
    """Return average product_resonance_score from fact_comments"""
    result = supabase.table("fact_comments").select("product_resonance_score").execute()
    if not result.data:
        return 0
    return sum(r["product_resonance_score"] for r in result.data) / len(result.data)

def get_video_quality_comments_ratio():
    result = supabase.table("fact_comments").select("product_resonance_score").execute()
    prs_values = [r["product_resonance_score"] for r in result.data if r["product_resonance_score"] is not None]
    if not prs_values:
        return 0
    quality_count = sum(1 for prs in prs_values if prs >= 0.3)
    return quality_count / len(prs_values)


def get_engagement_vs_prs():
    """Return DataFrame linking engagement_rate (dim_video) vs product_resonance_score (fact_comments)"""
    comments = supabase.table("fact_comments").select("videoid", "product_resonance_score").execute().data
    videos = supabase.table("dim_video").select("videoid", "engagement_rate").execute().data
    video_map = {v["videoid"]: v["engagement_rate"] for v in videos}
    
    data = []
    for c in comments:
        if c["videoid"] in video_map:
            data.append({
                "engagement_rate": video_map[c["videoid"]],
                "product_resonance_score": c["product_resonance_score"]
            })
    return pd.DataFrame(data)

def get_average_soe_metric_by_duration():
    videos = supabase.table("dim_video").select("videoid", "engagement_rate", "contentduration_seconds").execute().data
    comments = supabase.table("fact_comments").select("videoid", "product_resonance_score").execute().data
    df_videos = pd.DataFrame(videos)
    df_comments = pd.DataFrame(comments)
    df = df_videos.merge(df_comments, on="videoid")
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("contentduration_seconds")
          .agg(
              avg_engagement=("engagement_rate", "mean"),
              avg_prs=("product_resonance_score", "mean")
          )
          .reset_index()
          .sort_values("contentduration_seconds")
    )


def get_top_liked_product_category():
    """Return DataFrame of top 10 channels with sum of video likes broken down by cluster_label (5 clusters)."""
    
    allowed_clusters = [
        "cosmetics (eye & lip)",
        "hair treatment",
        "make up",
        "physical appearance",
        "skin care"
    ]
    
    videos = supabase.table("dim_video").select("videoid", "channelid", "likecount_video").execute().data
    df_videos = pd.DataFrame(videos)
    if df_videos.empty:
        return pd.DataFrame()
    
    top_channels = df_videos.groupby("channelid")["likecount_video"].sum().nlargest(10).index.tolist()
    df_videos_top = df_videos[df_videos["channelid"].isin(top_channels)]
    comments = supabase.table("fact_comments").select("videoid", "cluster_label").execute().data
    df_comments = pd.DataFrame(comments)
    
    if df_comments.empty:
        return pd.DataFrame()
    
    df = df_videos_top.merge(df_comments, on="videoid", how="left")
    df = df[df["cluster_label"].isin(allowed_clusters)]
    df_agg = df.groupby(["channelid", "cluster_label"])["likecount_video"].sum().reset_index()
    df_pivot = df_agg.pivot(index="channelid", columns="cluster_label", values="likecount_video").fillna(0)
    
    for cluster in allowed_clusters:
        if cluster not in df_pivot.columns:
            df_pivot[cluster] = 0
    
    df_pivot = df_pivot[allowed_clusters]
    df_pivot["total_likes"] = df_pivot.sum(axis=1)
    df_pivot = df_pivot.sort_values("total_likes", ascending=False).drop(columns="total_likes")
    
    return df_pivot.reset_index()


def get_video_topic_distribution():
    """
    Return top 5 topic categories by number of distinct videos from bridge_topic_category only.
    """
    bridge = supabase.table("bridge_topic_category").select("videoid", "topiccategoryid").execute().data
    if not bridge:
        return pd.DataFrame(columns=["topiccategoryid", "total_videos"])
    
    df_bridge = pd.DataFrame(bridge)

    topic_counts = (
        df_bridge.groupby("topiccategoryid")["videoid"]
        .nunique()
        .reset_index(name="total_videos")
    )

    top_topics = topic_counts.sort_values("total_videos", ascending=False).head(5).reset_index(drop=True)
    return top_topics

def get_total_comments_analyzed():
    result = supabase.table("fact_comments").select("commentid", count="exact").execute()
    return result.count

def get_comment_spam_ratio():
    comments = supabase.table("fact_comments").select("is_spam").execute().data
    if not comments:
        return 0
    spam_count = sum(1 for r in comments if r["is_spam"])
    return spam_count / len(comments)

def get_comment_quality_percentage():
    comments = supabase.table("fact_comments").select("sentiment_score").execute().data
    if not comments:
        return 0
    quality_count = sum(1 for r in comments if r["sentiment_score"] >= 0.7)
    return (quality_count / len(comments)) * 100

def get_comment_sentiment_ratio():
    comments = supabase.table("fact_comments").select("sentiment_label").execute().data
    if not comments:
        return {}
    df = pd.DataFrame(comments)
    counts = df["sentiment_label"].value_counts(normalize=True).to_dict()
    return counts

def get_top_trendy_hashtags(top_n=15):
    bridge = supabase.table("bridge_hashtag").select("hashtagid").execute().data
    hashtags = supabase.table("dim_hashtag").select("hashtagid", "hashtag").execute().data
    if not bridge or not hashtags:
        return pd.DataFrame()
    df_bridge = pd.DataFrame(bridge)
    df_hashtags = pd.DataFrame(hashtags)
    df = df_bridge.merge(df_hashtags, on="hashtagid")
    top_hashtags = df.groupby("hashtag").size().reset_index(name="count")
    return top_hashtags.sort_values("count", ascending=False).head(top_n)

def get_avg_prs_by_category():
    comments = supabase.table("fact_comments").select("cluster_label", "product_resonance_score").execute().data
    if not comments:
        return pd.DataFrame()
    df = pd.DataFrame(comments)
    return df.groupby("cluster_label").agg(avg_prs=("product_resonance_score", "mean")).reset_index().sort_values("avg_prs", ascending=False)

def get_category_sentiment_breakdown():
    comments = supabase.table("fact_comments").select("cluster_label", "sentiment_label").execute().data
    if not comments:
        return pd.DataFrame()
    df = pd.DataFrame(comments)
    return df.groupby(["cluster_label", "sentiment_label"]).size().reset_index(name="count")

def get_category_trend_distribution():
    comments = supabase.table("fact_comments").select("cluster_label", "dateid").execute().data
    dates = supabase.table("dim_date").select("dateid", "date").execute().data
    if not comments or not dates:
        return pd.DataFrame()
    df_comments = pd.DataFrame(comments)
    df_dates = pd.DataFrame(dates)
    df = df_comments.merge(df_dates, on="dateid")
    return df.groupby(["cluster_label", "date"]).size().reset_index(name="total_comments").sort_values(["cluster_label", "date"])
