import os
import re
import ast
import pandas as pd
from datetime import timedelta
from typing import List, Union
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from dotenv import load_dotenv
import emoji
from src.backend.spam_detector import SpamDetector
from src.backend.relevance_check import RelevanceChecker
from src.backend.sentiment_analysis import SentimentAnalyzer
from src.backend.compute_actionability import ActionabilityProcessor
from src.backend.prod_cat_clustering import ProductClustering
import numpy as np
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# ------------------ Helper Functions ------------------
PATTERNS = {
    "hashtag": re.compile(r"#(\w+)"),
    "emoji": re.compile(
        "[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]",
        flags=re.UNICODE
    ),
    "url": re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE),
    "mention": re.compile(r"@\w+", re.IGNORECASE),
    "special_char": re.compile(r"[^\w\s#]", re.UNICODE),
    "space": re.compile(r"\s+")
}

def get_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.query:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    if "shorts" in parsed.path:
        return parsed.path.split("/")[-1]
    match = re.search(r"([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def extract_hashtags(text) -> List[str]:
    """Extract hashtags directly from text, keep duplicates removed."""
    if not isinstance(text, str):
        return []
    hashtags = re.findall(r"#(\w+)", text)
    seen = set()
    clean_tags = []
    for tag in hashtags:
        tag = tag.strip()
        if tag and tag not in seen:
            clean_tags.append(tag)
            seen.add(tag)
    return clean_tags

def extract_emojis(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    emojis_found = list(set(PATTERNS["emoji"].findall(text)))
    return [emoji.demojize(e, delimiters=(" ", " ")).strip() for e in emojis_found]

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = PATTERNS["url"].sub(" ", text)
    text = PATTERNS["mention"].sub(" ", text)
    text = PATTERNS["special_char"].sub(" ", text)
    text = PATTERNS["space"].sub(" ", text).strip()
    text = PATTERNS["hashtag"].sub(" ", text)
    text = PATTERNS["emoji"].sub(" ", text)
    text = " ".join(w for w in text.split() if re.fullmatch(r"[a-z0-9]+", w))
    return text

def iso8601_to_seconds(duration: str) -> int:
    if not isinstance(duration, str) or not duration.startswith("P"):
        return None
    time_str = duration.replace("P", "")
    days, hours, minutes, seconds = 0, 0, 0, 0
    day_match = re.search(r"(\d+)D", time_str)
    if day_match:
        days = int(day_match.group(1))
    time_part = time_str.split("T")[-1] if "T" in time_str else time_str
    hour_match = re.search(r"(\d+)H", time_part)
    minute_match = re.search(r"(\d+)M", time_part)
    second_match = re.search(r"(\d+)S", time_part)
    if hour_match:
        hours = int(hour_match.group(1))
    if minute_match:
        minutes = int(minute_match.group(1))
    if second_match:
        seconds = int(second_match.group(1))
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds).total_seconds()

def extract_topic_categories(cat_list) -> List[str]:
    if isinstance(cat_list, list):
        cats = cat_list
    elif isinstance(cat_list, str):
        try:
            cats = ast.literal_eval(cat_list) if cat_list.startswith("[") else [cat_list]
        except Exception:
            cats = [cat_list]
    else:
        return []
    clean_cats = [c.split("wiki/")[-1] for c in cats if isinstance(c, str) and "wiki/" in c]
    return clean_cats

def calculate_product_resonance_score(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    
    # 1. Relevance (R)
    R = df_copy['weighted_relevance']

    # 2. Actionability (A)
    max_actionability = df_copy['actionability_score'].max()
    A = df_copy['actionability_score'] / max_actionability if max_actionability > 0 else 0

    # 3. Sentiment (S)
    conditions = [
        df_copy['sentiment_label'] == 'POSITIVE',
        df_copy['sentiment_label'] == 'NEGATIVE'
    ]
    choices = [1, -1]
    polarity_multiplier = np.select(conditions, choices, default=0)
    polarity_score = polarity_multiplier * df_copy['sentiment_score']
    S = (polarity_score + 1) / 2

    # 4. Influence (I)
    avg_likes_df = df_copy.groupby('videoId').agg(avg_likes=('commentLikes', 'mean')).reset_index()
    df_copy = pd.merge(df_copy, avg_likes_df, on='videoId', how='left')
    raw_influence = (df_copy['commentLikes'] / df_copy['avg_likes']).fillna(0)
    I = np.clip(raw_influence / 4.0, 0, 1)
    df_copy.drop(columns=['avg_likes'], inplace=True)
    
    df_copy['product_resonance_score'] = (0.40 * R) + (0.25 * A) + (0.20 * S) + (0.15 * I)

    return df_copy

def calculate_product_resonance_score(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    R = df_copy['weighted_relevance']
    max_actionability = df_copy['actionability_score'].max()
    A = df_copy['actionability_score'] / max_actionability if max_actionability > 0 else 0
    conditions = [
        df_copy['sentiment_label'] == 'POSITIVE',
        df_copy['sentiment_label'] == 'NEGATIVE'
    ]
    choices = [1, -1]
    polarity_multiplier = np.select(conditions, choices, default=0)
    polarity_score = polarity_multiplier * df_copy['sentiment_score']
    S = (polarity_score + 1) / 2
    avg_likes_df = df_copy.groupby('videoId').agg(avg_likes=('commentLikes', 'mean')).reset_index()
    df_copy = pd.merge(df_copy, avg_likes_df, on='videoId', how='left')
    raw_influence = (df_copy['commentLikes'] / df_copy['avg_likes']).fillna(0)
    I = np.clip(raw_influence / 4.0, 0, 1)
    df_copy.drop(columns=['avg_likes'], inplace=True)
    df_copy['product_resonance_score'] = (0.40 * R) + (0.25 * A) + (0.20 * S) + (0.15 * I)
    return df_copy

# ------------------ Scrape & Preprocess ------------------
def scrape_video(video_url: str) -> pd.DataFrame:
    youtube = build("youtube", "v3", developerKey=API_KEY)
    video_id = get_video_id(video_url)
    if not video_id:
        return pd.DataFrame()

    # --- Video details ---
    video_req = youtube.videos().list(
        part="snippet,statistics,contentDetails,topicDetails",
        id=video_id
    )
    video_res = video_req.execute()
    if not video_res["items"]:
        return pd.DataFrame()

    video = video_res["items"][0]
    topic_categories = video.get("topicDetails", {}).get("topicCategories", [])

    video_info = {
        "videoId": video_id,
        "title": video["snippet"]["title"],
        "description": video["snippet"]["description"],
        "publishedAt": video["snippet"]["publishedAt"],
        "channelId": video["snippet"]["channelId"],
        "tags": video["snippet"].get("tags", []),
        "duration": video["contentDetails"]["duration"],
        "viewCount": int(video["statistics"].get("viewCount", 0) or 0),
        "likeCount": int(video["statistics"].get("likeCount", 0) or 0),
        "commentCount": int(video["statistics"].get("commentCount", 0) or 0),
        "topicCategory": extract_topic_categories(topic_categories)
    }

    # --- Comments ---
    comments = []
    if video_info["commentCount"] > 0:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        )
        while request:
            response = request.execute()
            for item in response["items"]:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comment = {
                    **video_info,
                    "commentId": item["id"],
                    "commentText": snippet["textOriginal"],
                    "commentPublishedAt": snippet["publishedAt"],
                    "commentLikes": snippet["likeCount"]
                }
                comments.append(comment)

                # Replies
                if item["snippet"]["totalReplyCount"] > 0:
                    reply_request = youtube.comments().list(
                        part="snippet",
                        parentId=item["id"],
                        maxResults=100,
                        textFormat="plainText"
                    )
                    while reply_request:
                        reply_response = reply_request.execute()
                        for reply in reply_response["items"]:
                            reply_snippet = reply["snippet"]
                            reply_comment = {
                                **video_info,
                                "commentId": reply["id"],
                                "commentText": reply_snippet["textOriginal"],
                                "commentPublishedAt": reply_snippet["publishedAt"],
                                "commentLikes": reply_snippet["likeCount"]
                            }
                            comments.append(reply_comment)
                        reply_request = youtube.comments().list_next(reply_request, reply_response)

            request = youtube.commentThreads().list_next(request, response)

    df = pd.DataFrame(comments)
    if df.empty:
        df = pd.DataFrame([video_info])

    # ----------------- Preprocessing -----------------
    # Convert dates
    if "publishedAt" in df.columns:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce").dt.date
    if "commentPublishedAt" in df.columns:
        df["commentPublishedAt"] = pd.to_datetime(df["commentPublishedAt"], errors="coerce").dt.date

    # Convert duration
    if "duration" in df.columns:
        df["contentDuration_seconds"] = df["duration"].apply(iso8601_to_seconds)

    # Ensure numeric columns
    for col in ["viewCount", "likeCount", "commentCount", "commentLikes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Extract hashtags and emojis from title + description + tags
    combined_text = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["tags"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    )
    df["hashtags_video"] = combined_text.apply(extract_hashtags)
    df["emojis_video"] = combined_text.apply(extract_emojis)
    df["video_text"] = combined_text.apply(clean_text)
    df["engagement_rate"] = (df["likeCount"] + df["commentCount"]) / df["viewCount"].replace(0, 1)

    # ----------------- Spam Detection -----------------
    if not df.empty and 'commentText' in df.columns:
        spam_detector = SpamDetector()
        df = spam_detector.spam_detector_pipeline(df, text_col="commentText")

    # ----------------- Relevance Check -----------------
    if not df.empty and 'commentText' in df.columns:
        relevance_checker = RelevanceChecker()
        df = relevance_checker.preprocess_comments(df, text_col="commentText")
        df = relevance_checker.relevance_check_pipeline(df)

    # ----------------- Sentiment Analysis -----------------
    if not df.empty and 'commentText' in df.columns:
        sentiment_analyzer = SentimentAnalyzer()
        df = sentiment_analyzer.sentiment_pipeline(df, text_column="commentText")

    # ----------------- Actionability Analysis -----------------
    if not df.empty and 'commentText' in df.columns:
        actionability_processor = ActionabilityProcessor()
        df = actionability_processor.actionability_pipeline(df, text_column="commentText")

    # ----------------- Product Clustering -----------------
    if not df.empty and 'commentText' in df.columns:
        product_clustering = ProductClustering(df)
        df = product_clustering.clustering_pipeline()

    # ----------------- Product Resonance Score -----------------
    if not df.empty and 'product_resonance_score' not in df.columns:
        df = calculate_product_resonance_score(df)
        
    cols_to_drop = [
    "duration",
    "video_text",
    "comment_clean",
    "text_cleaned",
    "textcleaned",
    "too_short"
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df