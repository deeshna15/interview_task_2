import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Page Config
st.set_page_config(
    page_title="Universal Memory Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Dark theme aesthetic styling */
    .stApp {
        background-color: #0E1117;
        color: #E0E2E6;
    }
    
    /* Custom headers */
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8F8F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Subheaders */
    .sub-header {
        color: #8A9AAD;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    
    /* Summary thread section highlighting */
    .summary-thread-card {
        background: rgba(255, 75, 75, 0.1);
        border: 1px solid rgba(255, 75, 75, 0.3);
        border-radius: 10px;
        padding: 12px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    /* Chat message bubble improvements */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    
    /* Styling for the memory indicator */
    .memory-badge {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px dashed #38BDF8;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None
if "provider" not in st.session_state:
    st.session_state.provider = "gemini"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# API interaction helpers
def fetch_threads():
    try:
        response = requests.get(f"{API_URL}/threads")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
    return []

def create_thread(title, is_summary=False):
    try:
        response = requests.post(f"{API_URL}/threads", json={"title": title, "is_summary_thread": is_summary})
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error creating thread: {e}")
    return None

def delete_thread(thread_id):
    try:
        response = requests.delete(f"{API_URL}/{thread_id}" if "threads" not in API_URL else f"{API_URL}/{thread_id}")
        # Just in case API endpoint is /threads/{thread_id}
        response = requests.delete(f"{API_URL}/threads/{thread_id}")
        if response.status_code == 204:
            return True
    except Exception as e:
        st.error(f"Error deleting thread: {e}")
    return False

def fetch_messages(thread_id):
    try:
        response = requests.get(f"{API_URL}/threads/{thread_id}/messages")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching messages: {e}")
    return []

def send_message(thread_id, content, api_key, provider):
    try:
        payload = {"content": content, "api_key": api_key, "provider": provider}
        response = requests.post(f"{API_URL}/threads/{thread_id}/messages", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            # Parse error detail
            err_data = response.json()
            st.error(f"Error: {err_data.get('detail', 'Unknown error occurred')}")
    except Exception as e:
        st.error(f"Error sending message: {e}")
    return None

def regenerate_summary(thread_id, api_key, provider):
    try:
        payload = {"content": "Regenerate Summary", "api_key": api_key, "provider": provider}
        response = requests.post(f"{API_URL}/threads/{thread_id}/regenerate-summary", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            err_data = response.json()
            st.error(f"Error: {err_data.get('detail', 'Unknown error occurred')}")
    except Exception as e:
        st.error(f"Error regenerating summary: {e}")
    return None

# Load threads
threads = fetch_threads()

# Auto-initialize database with default threads (Thread 1, Thread 2, and Summary Thread 3) if empty
if not threads:
    with st.spinner("Initializing default chat threads..."):
        t1 = create_thread("Thread 1: Travel Plan")
        t2 = create_thread("Thread 2: Coding Practice")
        t3 = create_thread("Thread 3: Summary of all Threads", is_summary=True)
        
        # Load sample conversations so the memory has initial content
        if t1:
            send_message(t1["id"], "I am planning a trip to Japan next winter. What are some top places to visit?", os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"), "gemini")
        if t2:
            send_message(t2["id"], "I am writing a Python program using FastAPI. What are the best practices for structuring API endpoints?", os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"), "gemini")
            
        threads = fetch_threads()
        if threads:
            st.session_state.active_thread_id = threads[0]["id"]
            st.rerun()

# Set initial active thread if not set
if st.session_state.active_thread_id is None and threads:
    st.session_state.active_thread_id = threads[0]["id"]

# Find current thread object
current_thread = next((t for t in threads if t["id"] == st.session_state.active_thread_id), None)
if not current_thread and threads:
    current_thread = threads[0]
    st.session_state.active_thread_id = current_thread["id"]


# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🧠 Universal Mind AI</div>", unsafe_allow_html=True)
    st.caption("FastAPI + Streamlit Chat with Cross-Thread Memory")
    
    st.divider()
    
    # LLM Settings
    st.markdown("### ⚙️ LLM Configuration")
    st.session_state.provider = st.selectbox(
        "AI Provider",
        options=["gemini", "openai"],
        index=0 if st.session_state.provider == "gemini" else 1,
        key="provider_select"
    )
    
    # Pre-populate key from environment variables if present
    default_key = ""
    if st.session_state.provider == "gemini":
        default_key = os.getenv("GEMINI_API_KEY", "")
    elif st.session_state.provider == "openai":
        default_key = os.getenv("OPENAI_API_KEY", "")
        
    st.session_state.api_key = st.text_input(
        f"{st.session_state.provider.capitalize()} API Key",
        type="password",
        value=st.session_state.api_key or default_key,
        placeholder="Enter API Key here..."
    )
    
    st.divider()
    
    # Thread Management
    st.markdown("### 💬 Active Conversations")
    
    # Create thread input
    with st.form("create_thread_form", clear_on_submit=True):
        new_title = st.text_input("New Thread Name", placeholder="e.g. Shopping List")
        is_summary = st.checkbox("Is Summary Thread?", help="Check this to make it a summary synthesizing all other threads.")
        submit_btn = st.form_submit_button("➕ Create Thread", use_container_width=True)
        if submit_btn and new_title.strip():
            new_t = create_thread(new_title, is_summary)
            if new_t:
                st.session_state.active_thread_id = new_t["id"]
                st.toast(f"Created thread: {new_title}")
                st.rerun()

    st.divider()

    # Normal Threads List
    st.markdown("#### Regular Threads")
    normal_threads = [t for t in threads if not t["is_summary_thread"]]
    for t in normal_threads:
        col1, col2 = st.columns([4, 1])
        with col1:
            is_active = (t["id"] == st.session_state.active_thread_id)
            btn_label = f"💬 {t['title']}"
            if is_active:
                btn_label = f"👉 {t['title']} (Active)"
            
            # Clicking a thread makes it active
            if st.button(btn_label, key=f"select_{t['id']}", use_container_width=True):
                st.session_state.active_thread_id = t["id"]
                st.rerun()
        with col2:
            # Delete button
            if st.button("🗑️", key=f"del_{t['id']}", help="Delete this thread"):
                if delete_thread(t["id"]):
                    st.toast(f"Deleted thread {t['title']}")
                    st.session_state.active_thread_id = None
                    st.rerun()

    # Summary Threads List (Highlighted)
    summary_threads = [t for t in threads if t["is_summary_thread"]]
    if summary_threads:
        st.divider()
        st.markdown("#### Summary Threads")
        for t in summary_threads:
            st.markdown(f"<div class='summary-thread-card'>", unsafe_allow_html=True)
            col1, col2 = st.columns([4, 1])
            with col1:
                is_active = (t["id"] == st.session_state.active_thread_id)
                btn_label = f"📋 {t['title']}"
                if is_active:
                    btn_label = f"⚡ {t['title']} (Active)"
                if st.button(btn_label, key=f"select_{t['id']}", use_container_width=True):
                    st.session_state.active_thread_id = t["id"]
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{t['id']}", help="Delete summary thread"):
                    if delete_thread(t["id"]):
                        st.toast(f"Deleted summary thread")
                        st.session_state.active_thread_id = None
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# ================= MAIN CHAT AREA =================
if current_thread:
    # Header with title
    st.markdown(f"<div class='main-header'>{current_thread['title']}</div>", unsafe_allow_html=True)
    if current_thread['is_summary_thread']:
        st.markdown("<div class='sub-header'>A special synthesis thread displaying automatically updated takeaways across all discussions.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='sub-header'>Chat freely. The AI remembers context from your other threads automatically.</div>", unsafe_allow_html=True)

    # Inform user about API Key requirement if not set
    if not st.session_state.api_key:
        st.warning("⚠️ No API Key set. Please enter your Gemini or OpenAI API Key in the sidebar to start chatting.", icon="🔑")

    # Fetch and display chat messages
    messages = fetch_messages(current_thread["id"])
    
    # For Summary Thread, offer a manual regenerate button
    if current_thread["is_summary_thread"]:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Regenerate Global Summary", use_container_width=True, type="primary"):
                with st.spinner("Synthesizing summary from all conversations..."):
                    res = regenerate_summary(current_thread["id"], st.session_state.api_key, st.session_state.provider)
                    if res:
                        st.toast("Summary regenerated successfully!")
                        st.rerun()

    # Display conversations
    for msg in messages:
        role = "user" if msg["sender"] == "user" else "assistant"
        with st.chat_message(role):
            # Universal memory indicator for Assistant responses in normal threads
            if role == "assistant" and not current_thread["is_summary_thread"]:
                # Check if other threads summaries exist to display a badge
                other_summaries = [t for t in threads if t["id"] != current_thread["id"] and t["summary"] and not t["is_summary_thread"]]
                if other_summaries:
                    st.markdown("<span class='memory-badge'>🧠 Universal Memory Engaged</span>", unsafe_allow_html=True)
            st.write(msg["content"])

    # Chat Input
    if not current_thread["is_summary_thread"]:
        # Standard Chat Input for normal threads
        user_input = st.chat_input("Message the AI...")
        if user_input:
            if not st.session_state.api_key:
                st.error("Please configure your API key in the sidebar before sending messages.")
            else:
                # Display user message immediately
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Fetch AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = send_message(
                            thread_id=current_thread["id"],
                            content=user_input,
                            api_key=st.session_state.api_key,
                            provider=st.session_state.provider
                        )
                        if response:
                            st.write(response["content"])
                            st.rerun()
    else:
        # Prompt user to ask questions about other threads
        user_input = st.chat_input("Ask a question about your threads (e.g. 'Summarize what I discussed about coding')")
        if user_input:
            if not st.session_state.api_key:
                st.error("Please configure your API key in the sidebar before sending messages.")
            else:
                with st.chat_message("user"):
                    st.write(user_input)
                
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing thread history..."):
                        response = send_message(
                            thread_id=current_thread["id"],
                            content=user_input,
                            api_key=st.session_state.api_key,
                            provider=st.session_state.provider
                        )
                        if response:
                            st.write(response["content"])
                            st.rerun()

else:
    st.markdown("<div class='main-header'>Welcome to Universal Mind AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Select or create a conversation thread in the sidebar to start.</div>", unsafe_allow_html=True)
