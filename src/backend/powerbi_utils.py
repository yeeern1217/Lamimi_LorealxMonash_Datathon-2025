import pandas as pd
import os
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
from .database import (
    get_total_video_analyzed,
    get_average_engagement_rate,
    get_average_prs_score,
    get_video_quality_comments_ratio,
    get_engagement_vs_prs,
    get_average_soe_metric_by_duration,
    get_top_liked_product_category,
    get_video_topic_distribution,
    get_total_comments_analyzed,
    get_comment_spam_ratio,
    get_comment_quality_percentage,
    get_comment_sentiment_ratio,
    get_top_trendy_hashtags,
    get_avg_prs_by_category,
    get_category_sentiment_breakdown,
    get_category_trend_distribution
)

load_dotenv()
# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def render_powerbi():
    powerbi_html = """
    <div class="iframe-box">
        <iframe 
            title="PowerBI Report"
            width="100%" 
            height="1000" 
            src="https://app.powerbi.com/view?r=eyJrIjoiY2FjYzU4ZWYtYjZmMy00YzY2LWI2YzYtMzAyYWE2YTA5NDAxIiwidCI6ImE2M2JiMWE5LTQ4YzItNDQ4Yi04NjkzLTMzMTdiMDBjYTdmYiIsImMiOjEwfQ%3D%3D"
            frameborder="0" 
            allowFullScreen="true">
        </iframe>
    </div>
    <style>
        .iframe-box {
            border: 1px solid #2E4057;
            border-radius: 12px;
            padding: 10px;
            background-color: #ffffff;
            box-shadow: 0px 8px 15px rgba(46, 64, 87, 0.2);
            margin: 20px 0;
        }
    </style>
    """
    st.markdown(powerbi_html, unsafe_allow_html=True)

def fetch_chart_data(chart_name: str):
    """
    Map chart name to corresponding database function and return the data.
    """
    chart_map = {
        "KPI: Total Video Analyzed": get_total_video_analyzed,
        "KPI: Average Engagement Rate": get_average_engagement_rate,
        "KPI: Average PRS Score": get_average_prs_score,
        "KPI: Video Quality Comments Ratio": get_video_quality_comments_ratio,
        "Chart: Engagement Rate VS Product Resonance Score": get_engagement_vs_prs,
        "Chart: Average SoE Metric by Content Duration": get_average_soe_metric_by_duration,
        "Chart: Top Liked Product Category Content": get_top_liked_product_category,
        "Chart: Video Topic Category Distribution": get_video_topic_distribution,
        "KPI: Total Comments Analyzed": get_total_comments_analyzed,
        "KPI: Comment Spam Ratio": get_comment_spam_ratio,
        "KPI: Comment Quality (%)": get_comment_quality_percentage,
        "KPI: Comments Sentiment Ratio": get_comment_sentiment_ratio,
        "Chart: Top 15 Most Trendy Hashtags": get_top_trendy_hashtags,
        "Chart: Average PRS by Product Category": get_avg_prs_by_category,
        "Chart: Product Category Sentiment Breakdown": get_category_sentiment_breakdown,
        "Chart: Product Category Trend Distribution": get_category_trend_distribution
    }

    if chart_name in chart_map:
        func = chart_map[chart_name]
        print(f"[DEBUG] Fetching data for '{chart_name}' using function: {func.__name__}")
        data = func()
        print(f"[DEBUG] Data retrieved for '{chart_name}':\n{data if isinstance(data, pd.DataFrame) else str(data)}")
        return data
    else:
        print(f"[WARNING] No mapping found for chart: {chart_name}")
        return None
    
def generate_insights(user_input: str, chart_name: str):
    data = fetch_chart_data(chart_name)
    
    if isinstance(data, pd.DataFrame):
        data_summary = data.head(10).to_string(index=False)
    else:
        data_summary = str(data)
    
    system_prompt = f"""
    You are a professional business insights assistant for L’Oréal marketing team.
    You are analyzing dashboards with KPIs and charts. Use the following guidelines:

    1. Use the data provided below to generate insights.
    2. Explain the charts/metrics in plain, simple language anyone can understand.
    3. Provide specific actionable insights and recommendations for the marketing team based on the data.
    4. Avoid jargon unless necessary. If you use a technical term (e.g., “sentiment score”), define it simply.
    5. Always structure your response clearly as:
    - 📊 Explanation
    - 🔍 Insights
    - ✅ Recommendations
    6. Keep answers clear, concise, and ideally under 200 words.
    7. Ensure the response is complete and self-contained.

    Chart/Metric Name: {chart_name}

    Data:
    {data_summary}
    """
    user_prompt = f"{user_input}\nPlease base your response on the above data."

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
        max_tokens=700
    )

    response_text = response.choices[0].message.content
    return response_text
