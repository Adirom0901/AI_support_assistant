# frontend.py - Streamlit UI that talks to the FastAPI backend
import os
import uuid
 
import requests
import streamlit as st
 
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
 
st.set_page_config(page_title="Order Support Assistant", page_icon="🛒", layout="centered")
 
 
def new_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
 
 
if "thread_id" not in st.session_state:
    new_chat()
 
 
@st.cache_data(ttl=10)
def api_online() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).ok
    except requests.RequestException:
        return False
 
 
def ask_backend(message: str) -> str:
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    resp = requests.post(
        f"{API_URL}/chat",
        json={"message": message, "thread_id": st.session_state.thread_id},
        headers=headers,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["reply"]
 
 
# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🛒 Support Assistant")
    st.caption("Ask about your order status, delivery date or tracking number.")
    st.button("➕ New chat", on_click=new_chat, use_container_width=True)
 
    st.divider()
    st.subheader("Try these")
    examples = [
        "Where is my order ORD1002?",
        "When will ORD1001 be delivered?",
        "What is the tracking number for ORD1004?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
 
    st.divider()
    st.write("Backend: " + ("🟢 online" if api_online() else "🔴 offline"))
 
# ---------------- Main chat area ----------------
st.title("Order Support Assistant")
 
if not st.session_state.messages:
    st.info("Hi! Share your order ID (for example **ORD1002**) and I'll look it up.")
 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
prompt = st.chat_input("Type your message...") or st.session_state.pop("pending", None)
 
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
 
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                reply = ask_backend(prompt)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            st.error(f"The server returned an error ({code}). Please try again.")
        except requests.RequestException:
            st.error("Could not reach the assistant backend. Is the API running?")