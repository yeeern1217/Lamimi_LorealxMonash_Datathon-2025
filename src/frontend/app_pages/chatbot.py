import streamlit as st
import time
import os
from src.backend.nl2sql_agentic import chat_with_db

def chatbot_page():
    # --- Load CSS ---
    css_path = os.path.join(os.path.dirname(__file__), "../styles/chatbot_page.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # --- Initialize session state ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # --- Page Header ---
    st.markdown(
        """
        <div class="chatbot-header">
            <h1>CommentSense Chat Assistant</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Display previous messages ---
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            avatar = "👤"
        elif message["role"] == "assistant":
            avatar = None  
        else:
            avatar = None

        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # --- Chat input ---
    if prompt := st.chat_input("Enter your query"):
        # Save user message
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate assistant response
        response = chat_with_db(prompt)

        # Typing effect with placeholder
        with st.chat_message("assistant"):
            placeholder = st.empty()
            typed_text = ""
            for char in response:
                typed_text += char
                placeholder.markdown(typed_text)
                time.sleep(0.005)

        # Save assistant response
        st.session_state["messages"].append({"role": "assistant", "content": response})
