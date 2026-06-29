# streamlit_viewer/app.py

import streamlit as st
import json
import glob
import os
from utils.chroma_connector import ChromaConnector

st.set_page_config(page_title="Dialogue Viewer", layout="wide")
st.title("🔍 RAG-DIVE Dialogue Viewer")

db_connector = ChromaConnector('./data/v_eval_filtered/')

def get_chunk_text_from_id(chunk_id):
    return db_connector.get_chunk_by_id(chunk_id)


# =========================================================
# Load JSONL files
# =========================================================
data_dir = st.sidebar.text_input("Data directory (absolute or relative)", value="./data")
if not os.path.isdir(data_dir):
    st.sidebar.error(f"Directory not found: `{data_dir}`")
    st.stop()

# Recursive search for .jsonl files
jsonl_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True))

if not jsonl_files:
    st.sidebar.warning(f"No .jsonl files found in `{data_dir}` (recursive search).")
    st.stop()

st.sidebar.success(f"Found {len(jsonl_files)} .jsonl file(s)")

# Show relative paths for readability
def display_name(path):
    return os.path.relpath(path, data_dir)

selected_file = st.sidebar.selectbox("File", jsonl_files, format_func=display_name)

conversations = []
with open(selected_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                conversations.append(json.loads(line))
            except json.JSONDecodeError as e:
                st.sidebar.warning(f"Skipped malformed line: {e}")

if not conversations:
    st.warning("File is empty or contains no valid JSON lines.")
    st.stop()

# =========================================================
# Auto-refresh for live monitoring
# =========================================================
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
if auto_refresh:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5000, key="refresh")

st.sidebar.markdown(f"**{len(conversations)} conversations loaded**")

# =========================================================
# Conversation selector
# =========================================================
if len(conversations) == 1:
    conv_idx = 0
    st.sidebar.info("1 conversation in file")
else:
    conv_idx = st.sidebar.slider("Conversation", 0, len(conversations) - 1, 0)

conv = conversations[conv_idx]

# =========================================================
# Header
# =========================================================

st.subheader("📚 Source Documents")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Article A**")
    st.info(conv.get("parent_doc_A", "?"))

with col2:
    st.markdown("**Article B**")
    st.info(conv.get("parent_doc_B", "?"))

role_text = conv.get("role", "")
if role_text:
    st.caption(f"**Persona:** {role_text[:200]}...")

st.divider()

# =========================================================
# Turns
# =========================================================
for turn in conv.get("conversation", []):
    turn_num = turn.get("turn_index", 0) + 1
    q_type = turn.get("type", "?")
    logic = turn.get("logic_type", "none")
    multi_hop = turn.get("multi_hop_flag", 0)
    bridging = turn.get("bridging_topic", None)

    hop_badge = "🟢 Multi-Hop" if multi_hop else "⚪ Single-Hop"
    type_color = {"Initial": "🔵", "Follow-up": "🟡", "Clarification": "🟠",
                  "Correction": "🔴", "Comparative": "🟣"}.get(q_type, "⚪")

    st.subheader(f"Turn {turn_num}  {type_color} {q_type} | {hop_badge} | Logic: `{logic}`")

    if bridging:
        st.caption(f"🔗 Bridging: {bridging}")

    with st.container():
        lcol, rcol = st.columns([1, 1])
        with lcol:
            st.markdown("**🧑 RAG Input (simulated user):**")
            st.info(turn.get("rag_input", ""))
            st.markdown("**📝 Standalone Question:**")
            st.caption(turn.get("question", ""))
        with rcol:
            st.markdown("**✅ Ground Truth:**")
            st.success(turn.get("answer", ""))

    st.markdown("**🤖 Target RAG Response:**")
    st.warning(turn.get("rag_answer", ""))

    ctx = turn.get("context", "")
    chunks = turn.get("ground_truth_chunks", [])

    with st.expander(f"📄 Retrieved Context & Ground Truth Chunks ({len(chunks)} chunks)"):
        # 1. Ground Truth Section (Live Fetch)
        if chunks:
            st.markdown("### 🎯 Ground Truth Source Chunks")
            for chunk_id in chunks:
                with st.expander(f"View content for: `{chunk_id}`"):
                    chunk_text = db_connector.get_chunk_by_id_for_streamlit(chunk_id)
                    st.text(chunk_text)
            st.divider()

        # 2. Retrieved Context Section
        st.markdown("### 🤖 Retrieved Context (by Target System)")
        if ctx:
            if isinstance(ctx, list):
                # Wenn es eine Liste ist, zeige die einzelnen Dokumente sauber an
                for idx, doc in enumerate(ctx):
                    with st.expander(f"Retrieved Document {idx + 1}"):
                        st.markdown(doc)
            else:
                st.code(str(ctx), language=None)
        else:
            st.caption("No context recorded.")