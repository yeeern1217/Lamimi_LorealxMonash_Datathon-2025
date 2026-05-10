import streamlit as st
from streamlit_option_menu import option_menu
from app_pages.chatbot import chatbot_page
from app_pages.dashboard import dashboard_page
from app_pages.analyzeComment import comment_senser_page    

# --- Page Config ---
st.set_page_config(
    page_title="L'ORÉAL CommentSense",
    layout="wide",
    menu_items={
        'About': "This is the website for L'ORÉAL Datathon."
    }
)

with st.sidebar:
    st.image("assets/logo-removebg.png", use_container_width=True)

    # Navigation menu
    selected = option_menu(
        menu_title="Main Menu",
        options=["Insights Hub", "Chat Assistant", "Comment Senser"],
        icons=["lightbulb", "robot", "chat-dots"],
        menu_icon="list",
        default_index=0,
    )

if selected == "Insights Hub":
    dashboard_page()
    
elif selected == "Chat Assistant":
    chatbot_page()
    
elif selected == "Comment Senser":
    comment_senser_page()


