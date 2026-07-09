import os
import sys
import sqlite3
import pickle
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Add workspace root to python path for robust importing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.advising_agent import run_advising_agent

# Set premium page layout
st.set_page_config(
    page_title="AcadAgent — IITGN BTech Academic Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling for UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF8C00 0%, #FFD700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #FF8C00;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: STATUS & SETTINGS ---
with st.sidebar:
    st.markdown("## ⚙️ Settings & Status")
    
    # 0. Clear Chat History
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": (
                    "Hello! I am **AcadAgent**, your IITGN BTech Academic Advisor. "
                    "I can help you construct minor pathways, check for timetable conflicts, "
                    "query details about courses, list prerequisites, and find academic policies. "
                    "\n\nWhat can I help you plan today?"
                )
            }
        ]
        st.rerun()
        
    st.markdown("---")
    
    # 1. Embedding Provider Selection (Toggles between Local and Google)
    st.markdown("### Embedding Provider")
    provider = st.selectbox(
        "Select Model Provider:",
        options=["local", "google"],
        index=0,
        help="Local uses sentence-transformers (MiniLM). Google uses Gemini Embeddings API (text-embedding-001)."
    )
    # Set the provider globally in environment variables
    os.environ["EMBEDDING_PROVIDER"] = provider
    
    st.markdown("---")
    st.markdown("### 📊 Database Statistics")
    
    # SQLite Stats
    db_file = "acadagent.db"
    if os.path.exists(db_file):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM courses;")
            course_count = cursor.fetchone()[0]
            conn.close()
            st.success(f"SQLite DB: {course_count} BTech courses loaded")
        except Exception as e:
            st.error(f"SQLite error: {e}")
    else:
        st.error("SQLite DB (acadagent.db) not found!")
        
    # NetworkX Stats
    graph_file = "graph/prerequisite_graph.pkl"
    if os.path.exists(graph_file):
        try:
            with open(graph_file, "rb") as f:
                G = pickle.load(f)
            st.success(f"NetworkX Graph: {G.number_of_nodes()} courses, {G.number_of_edges()} prereq links")
        except Exception as e:
            st.error(f"NetworkX error: {e}")
    else:
        st.error("Prerequisite graph not found!")
        
    # FAISS Index Status
    policy_dir = f"faiss_index_{provider}"
    plans_dir = f"faiss_index_course_plans_{provider}"
    # Fallback checks
    if provider == "google" and not os.path.exists(plans_dir):
        plans_dir = "faiss_index_course_plans_local" # fallback
        
    if os.path.exists(policy_dir):
        st.success(f"FAISS Policy Index: Ready ({provider})")
    else:
        st.error(f"FAISS Policy Index: Missing ({provider})")
        
    if os.path.exists(plans_dir):
        st.success(f"FAISS Course Plans Index: Ready")
    else:
        st.warning("FAISS Course Plans Index: Missing")
        
    st.markdown("---")
    st.markdown("### 💡 About AcadAgent")
    st.caption(
        "AcadAgent is an AI-powered academic planning agent for IITGN undergraduate students. "
        "It uses a structured SQLite database for timetables, a NetworkX graph for prerequisite topological sorting, "
        "and FAISS vector indexes for semantic policy and syllabus retrieval. "
        "Built strictly with LangGraph, SQLite, FAISS, and NetworkX."
    )

# --- MAIN APP VIEW ---
st.markdown("<div class='main-header'>AcadAgent 🎓</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>BTech Academic Advising Assistant — IIT Gandhinagar</div>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": (
                "Hello! I am **AcadAgent**, your IITGN BTech Academic Advisor. "
                "I can help you construct minor pathways, check for timetable conflicts, "
                "query details about courses, list prerequisites, and find academic policies. "
                "\n\nWhat can I help you plan today?"
            )
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if isinstance(content, list):
            # Unpack list content block in UI
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)
        st.markdown(content)

# Quick click sample queries
st.markdown("### 💡 Quick Queries")
col1, col2, col3, col4 = st.columns(4)

selected_query = None

with col1:
    if st.button("CS Minor Pathway (for Chemical)"):
        selected_query = "I am a Chemical Engineering student. I want to pursue a minor in Computer Science. Provide a pathway for it."
with col2:
    if st.button("Prerequisites for OS"):
        selected_query = "I want to take CS 330 (Operating Systems). Generate the sequence of courses I need to take first."
with col3:
    if st.button("Timetable Conflict Check"):
        selected_query = "Can I take Computing (ES 112) and Calculus (MA 103) in the same semester? Check for clashes."
with col4:
    if st.button("Tell me about Machine Learning"):
        selected_query = "I am planning to take ES 335 (Machine Learning) this semester. Tell me its details, prerequisites, and what is covered in it."

# Handle chat input or button click
user_query = st.chat_input("Ask AcadAgent a question...")
if selected_query:
    user_query = selected_query

if user_query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    # Store user query in history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Call advising agent
    with st.spinner("AcadAgent is analyzing databases, prerequisite graphs, and policies..."):
        try:
            # Pass conversation history (excluding the greeting message)
            chat_history = st.session_state.messages[1:-1]
            response = run_advising_agent(user_query, chat_history=chat_history)
            
            # Display agent response
            with st.chat_message("assistant"):
                st.markdown(response)
            # Store agent response in history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force refresh to display updated chat immediately (if triggered by button)
            if selected_query:
                st.rerun()
                
        except Exception as e:
            st.error(f"An error occurred while executing the agent: {e}")
