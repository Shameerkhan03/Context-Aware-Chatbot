import streamlit as st
from backend import get_answer

st.set_page_config(page_title="Knowledge Chatbot", layout="centered")

st.title("📚 Your Personal Knowledge Chatbot")
st.markdown("Ask questions about the uploaded blog posts!")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask something from your content:", key="user_input")

if query:
    with st.spinner("Thinking..."):
        # Pass st.session_state.history to the backend for conversational memory
        response = get_answer(query, st.session_state.history)
    
    st.session_state.history.append((query, response))
    
st.markdown("---")
st.subheader("Chat History")

for q, r in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {r}")

if st.button("Clear Chat"):
    st.session_state.history = []
    st.rerun()