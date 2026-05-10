import streamlit as st
import os
from src.backend.powerbi_utils import render_powerbi, generate_insights

def dashboard_page():
    # --- Load CSS ---
    css_path = os.path.join(os.path.dirname(__file__), "../styles/dashboard_page.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # --- Initialize session state ---
    if 'insights' not in st.session_state:
        st.session_state['insights'] = {}
    if 'selected_dashboard' not in st.session_state:
        st.session_state['selected_dashboard'] = "Share of Engagement (SoE) Dashboard"

    # --- Page Header ---
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>CommentSense Insights Hub</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Layout ---
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        render_powerbi()

    with col2:
        st.markdown('<div class="subheader-title"><h1>Insights Panel</h1></div>', unsafe_allow_html=True)
        dashboard_pages = ["Share of Engagement (SoE)", "Comment Analysis"]
        st.markdown('<div class="form-label"><h5>Select Dashboard</h5></div>', unsafe_allow_html=True)
        selected_dashboard = st.radio(
            "Select Dashboard",  # non-empty, will be hidden
            dashboard_pages,
            horizontal=True,
            label_visibility="collapsed"
        )

        st.session_state['selected_dashboard'] = selected_dashboard

        # --- KPI / Chart Options ---
        if "Share of Engagement" in selected_dashboard:
            options = [
                "KPI: Total Video Analyzed",
                "KPI: Average Engagement Rate",
                "KPI: Average PRS Score",
                "KPI: Video Quality Comments Ratio",
                "Chart: Engagement Rate VS Product Resonance Score",
                "Chart: Average SoE Metric by Content Duration",
                "Chart: Top Liked Product Category Content",
                "Chart: Video Topic Category Distribution"
            ]
        else:
            options = [
                "KPI: Total Comments Analyzed",
                "KPI: Comment Spam Ratio",
                "KPI: Comment Quality (%)",
                "KPI: Comments Sentiment Ratio",
                "Chart: Top 15 Most Trendy Hashtags",
                "Chart: Average PRS by Product Category",
                "Chart: Product Category Sentiment Breakdown",
                "Chart: Product Category Trend Distribution"
            ]

        # --- KPI / Chart Dropdown ---
        st.markdown('<div class="form-label"><h5>Choose KPI or Chart</h5></div>', unsafe_allow_html=True)
        selection = st.selectbox(
            "Choose KPI or Chart",  # non-empty, will be hidden
            options,
            label_visibility="collapsed"
        )
        # --- Button + Spinner ---
        if st.button("Generate Insights"):
            with st.spinner("Analyzing data and generating insights..."):
                response = generate_insights(
                    f"Provide insights for {selection} from {selected_dashboard}",
                    selection
                )
                st.session_state['insights'][selection] = response

        # --- Show Insights ---
        if selection in st.session_state['insights']:
            with st.expander("Insights", expanded=True):
                st.markdown(st.session_state['insights'][selection])

        st.markdown("</div>", unsafe_allow_html=True)
