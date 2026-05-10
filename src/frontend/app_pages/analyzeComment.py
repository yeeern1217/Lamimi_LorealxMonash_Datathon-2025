import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from src.backend.youtube_scrapper import scrape_video
from src.backend.report_generator import generate_pdf_report, generate_video_summary

def comment_senser_page():
    # --- Load CSS ---
    css_path = os.path.join(os.path.dirname(__file__), "../styles/analyze_comment_page.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # --- Header ---
    st.markdown(
        """
        <div class="analyze-header">
            <h1>Youtube Video & Comment Analysis</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Initialize session state ---
    for key, default in {
        "video_url": "",
        "video_data": None,
        "video_summary": None,
        "pdf_report": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Input field  ---
    st.markdown('<div class="form-label"><h5>Insert Youtube Video/Shorts URL</h5></div>', unsafe_allow_html=True)
    url_input = st.text_input(
        "",
        label_visibility="collapsed",
        value=st.session_state["video_url"],
        placeholder="Insert URL here and Enter (e.g., https://www.youtube.com/watch?v=example)",
    )

    # --- Processing function ---
    def process_video(url):
        df = scrape_video(url)
        if df.empty:
            st.error("No data found. Check the URL or API key.")
            return None, None, None

        video_title = df['title'].iloc[0]
        video_description = df['description'].iloc[0]
        summary = generate_video_summary(video_title, video_description)
        pdf_report = generate_pdf_report(df)
        return df, summary, pdf_report

    # --- Trigger when new URL entered ---
    if url_input and url_input != st.session_state.video_url:
        st.session_state.video_url = url_input
        with st.spinner("Scraping video and generating report..."):
            df, summary, pdf_report = process_video(url_input)
        st.session_state.video_data = df
        st.session_state.video_summary = summary
        st.session_state.pdf_report = pdf_report

        # --- Display if data exists ---
    if st.session_state.video_data is not None:
        df = st.session_state.video_data
        summary = st.session_state.video_summary
        pdf_report = st.session_state.pdf_report
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="subheader-title"><h1>Video Details</h1></div>', unsafe_allow_html=True)
        url = st.session_state.video_url
        video_id = None
        patterns = [
            r"v=([a-zA-Z0-9_-]{11})",   # standard watch?v=
            r"youtu\.be/([a-zA-Z0-9_-]{11})",  # short link
            r"shorts/([a-zA-Z0-9_-]{11})"      # shorts
        ]
        for p in patterns:
            match = re.search(p, url)
            if match:
                video_id = match.group(1)
                break

        # --- Video details ---
        video_title = df['title'].iloc[0]
        video_description = df['description'].iloc[0]
        video_channel_id = df['channelId'].iloc[0]
        video_published_date = df['publishedAt'].iloc[0]
        video_likes = df['likeCount'].iloc[0]
        video_views = df['viewCount'].iloc[0]
        video_comments = df['commentCount'].iloc[0]
        video_engagement_rate = df['engagement_rate'].iloc[0]

        st.markdown(
            f"""
            <div class="video-card">
                <div class="video-wrapper">
                    <iframe src="https://www.youtube.com/embed/{video_id}" 
                    frameborder="0" allowfullscreen></iframe>
                </div>
                <div class="video-detail-box">
                    <p class="video-title">Title: {video_title}</p>
                    <p class="video-description">{video_description}</p>
                    <div class="summary"><b>Summary:</b> {summary}</div>
                    <div class="meta" style="display: flex; gap: 2rem;">
                        <div class="meta-column" style="flex: 1;">
                            <p><b>Channel ID:</b> {video_channel_id}</p>
                            <p><b>Published Date:</b> {video_published_date}</p>
                            <p><b>Engagement Rate:</b> {video_engagement_rate:.2%}</p>
                        </div>
                        <div class="meta-column" style="flex: 1;">
                            <p><b>Likes:</b> {video_likes:,}</p>
                            <p><b>Views:</b> {video_views:,}</p>
                            <p><b>Comments:</b> {video_comments:,}</p>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()
        # --- Metrics calculation ---
        total_comments = len(df)
        spam_count = df['is_spam'].sum()
        non_spam_count = total_comments - spam_count

        # Non-spam : spam ratio in percentage
        non_spam_percent = (non_spam_count / total_comments) * 100 if total_comments > 0 else 0
        spam_percent = (spam_count / total_comments) * 100 if total_comments > 0 else 0
        nonspam_spam_ratio = f"{non_spam_percent:.0f} : {spam_percent:.0f}"

        relevant_comments = df['is_relevant'].sum()
        relevance_percent = (relevant_comments / total_comments) * 100 if total_comments > 0 else 0
        avg_product_resonance = df['product_resonance_score'].mean() if total_comments > 0 else 0

        # --- Custom Metric Cards ---
        st.markdown(
            f"""
            <div class="subheader-title"><h1>Comment Analysis Overview</h1></div>
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-title">Total Comments</div>
                    <div class="metric-value">{total_comments:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Non-Spam : Spam Ratio</div>
                    <div class="metric-value">{nonspam_spam_ratio}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Comment Relevance %</div>
                    <div class="metric-value">{relevance_percent:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Average Product Resonance Score</div>
                    <div class="metric-value">{avg_product_resonance:.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        # --- Chart 1: Sentiment Distribution by Cluster ---
        pivot = df.groupby(['cluster_label', 'sentiment_label']).size().unstack(fill_value=0)
        pivot_percent = pivot.div(pivot.sum(axis=1), axis=0)
        color_map = {'POSITIVE': 'green', 'NEGATIVE': 'red'}
        pivot_percent = pivot_percent[[col for col in ['POSITIVE','NEGATIVE'] if col in pivot_percent.columns]]
        fig1, ax1 = plt.subplots(figsize=(3.5,2.5))
        pivot_percent.plot(kind='area', stacked=True, ax=ax1, color=[color_map[c] for c in pivot_percent.columns])
        ax1.set_ylabel("Percentage", fontsize=8)
        ax1.set_xlabel("Product Cluster", fontsize=8)
        ax1.tick_params(axis='both', labelsize=7)
        ax1.legend(title="Sentiment", fontsize=7, title_fontsize=8)
        row1_col1.markdown('<div class="chart-title"><h3>Sentiment Distribution by Cluster</h3></div>', unsafe_allow_html=True)
        row1_col1.pyplot(fig1)
        plt.tight_layout()
        
        # --- Chart 2: Actionability Label Distribution ---
        counts = df['actionability_label'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(3.5,2.5))  # smaller
        colors = sns.color_palette('pastel')[0:len(counts)]
        explode = [0.05]*len(counts)
        ax2.pie(
            counts, labels=counts.index, autopct='%1.1f%%', colors=colors,
            explode=explode, shadow=True, startangle=90, textprops={'fontsize':8}
        )
        ax2.axis('equal')
        row1_col2.markdown('<div class="chart-title"><h3>Actionability Label Distribution</h3></div>', unsafe_allow_html=True)
        row1_col2.pyplot(fig2)

        # --- Chart 3: Product Resonance Distribution ---
        fig3, ax3 = plt.subplots(figsize=(3.5,2.5))  # smaller
        sns.histplot(
            data=df, x='product_resonance_score', hue='cluster_label',
            multiple='stack', palette='Set2', ax=ax3
        )
        ax3.set_xlabel("Product Resonance Score", fontsize=8)
        ax3.set_ylabel("Number of Comments", fontsize=8)
        ax3.tick_params(axis='both', labelsize=7)
        row2_col1.markdown('<div class="chart-title"><h3>Product Resonance Distribution</h3></div>', unsafe_allow_html=True)
        row2_col1.pyplot(fig3)

        # --- Chart 4: Scatter Plot (Weighted Relevance vs Sentiment Score) ---
        fig4, ax4 = plt.subplots(figsize=(3.5,2.5))  # smaller
        sns.scatterplot(
            data=df, x='weighted_relevance', y='sentiment_score',
            hue='cluster_label', palette='Set1', ax=ax4, s=30, alpha=0.7  # smaller markers
        )
        ax4.set_xlabel("Weighted Relevance", fontsize=8)
        ax4.set_ylabel("Sentiment Score", fontsize=8)
        ax4.tick_params(axis='both', labelsize=7)
        row2_col2.markdown('<div class="chart-title"><h3>Weighted Relevance vs Sentiment Score</h3></div>', unsafe_allow_html=True)
        row2_col2.pyplot(fig4)
        
        row2_col1.markdown('<div class="chart-title"><h3>Comment Analysis Table</h3></div>', unsafe_allow_html=True)
        cols_to_drop = [
        "videoId",
        "title",
        "description",
        "publishedAt",
        "channelId",
        "tags",
        "viewCount",
        "liekCount",
        "commentCount",
        "topicCategory",
        "contentDuration_seconds",
        "hashtags_video",
        "likeCount"
        "emojis_video",
        ]
        df_comment = df.drop(columns=cols_to_drop, errors="ignore")
        st.dataframe(df_comment)
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
        st.divider()
        # === Download button ===
        file_path = r"C:\UM\Comp\CommentSense Report-The best of L’Oreal.pdf"

        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="Download Report",
            data=pdf_bytes,
            file_name="CommentSense_Report-The_best_of_L_Oreal.pdf",
            mime="application/pdf"
        )