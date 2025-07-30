import streamlit as st
from chatbot import get_response

# 1. Basic page config
st.set_page_config(page_title="FAQ Chatbot", layout="centered")

st.title("📘 FAQ Chatbot")
st.write("Ask any question from our FAQ:")

# 2. Input box
user_question = st.text_input("Your question here:")

# 3. When they ask, call the chain
if user_question:
    with st.spinner("Thinking…"):
        answer = get_response(user_question)
    st.markdown(f"**Answer:** {answer}")
