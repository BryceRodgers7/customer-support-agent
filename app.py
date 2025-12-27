# app.py
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from support_bot import CustomerSupportBot, SupportSession

load_dotenv()  # optional; reads OPENAI_API_KEY from .env

st.set_page_config(page_title="Customer Support Agent Demo", page_icon="💬", layout="centered")
st.title("💬 Customer Support Agent (Demo)")
st.caption("Agentic support bot with tool-calling (products, orders, returns, ticketing).")

# Create bot once per session
if "bot" not in st.session_state:
    st.session_state.bot = CustomerSupportBot(model="gpt-4o")

# Maintain a session object for conversation continuity
if "support_session" not in st.session_state:
    st.session_state.support_session = SupportSession()

# Store chat history for UI rendering
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! How can I help today? You can ask about products, order status (e.g., A1001), returns, or troubleshooting.",
        }
    ]

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a support question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = st.session_state.bot.reply(st.session_state.support_session, prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("Demo controls")
    st.write("Try:")
    st.code("Do you have any ANC headphones in stock?", language="text")
    st.code("Where is my order A1001?", language="text")
    st.code("What's the return policy for CAM4K-003?", language="text")
    st.code("My StreamCam won’t autofocus. Can you troubleshoot?", language="text")

    if st.button("Reset chat"):
        st.session_state.messages = st.session_state.messages[:1]
        st.session_state.support_session = SupportSession()
        st.rerun()
