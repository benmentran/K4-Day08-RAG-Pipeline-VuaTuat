"""
IELTS Writing Assistant — Editorial Premium UI
Inspired by Later.com: clean editorial layout, generous whitespace,
bento grid for stats, serif headlines, refined typography hierarchy,
warm cream/off-white palette with deep ink text.

Connects to real RAG pipeline (Task 4-10) when available, with
graceful fallback when modules are missing.
"""

import streamlit as st
from datetime import datetime
from pathlib import Path

# Load .env first
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="IELTS Writing Assistant — Band 8.0+ Strategy",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# PIPELINE IMPORTS (graceful fallback)
# =============================================================================

PIPELINE_STATUS = {
    "task4_chunking": False,
    "task5_semantic": False,
    "task6_lexical": False,
    "task7_reranking": False,
    "task8_pageindex": False,
    "task9_retrieval": False,
    "task10_generation": False,
}

DOC_STATS = {
    "total_docs": 13,
    "total_chunks": 0,
}

generate_with_citation = None
retrieve = None
semantic_search = None
lexical_search = None

try:
    from src.task5_semantic_search import semantic_search as _sem
    semantic_search = _sem
    PIPELINE_STATUS["task5_semantic"] = True
except ImportError:
    pass

try:
    from src.task7_reranking import rerank_rrf
    PIPELINE_STATUS["task7_reranking"] = True
except ImportError:
    pass

try:
    from src.task6_lexical_search import lexical_search as _lex
    lexical_search = _lex
    PIPELINE_STATUS["task6_lexical"] = True
except ImportError:
    pass

try:
    from src.task4_chunking_indexing import COLLECTION_NAME, CHROMA_DIR
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
        DOC_STATS["total_chunks"] = collection.count()
        PIPELINE_STATUS["task4_chunking"] = True
    except Exception:
        pass
except ImportError:
    pass

try:
    from src.task8_pageindex_vectorless import pageindex_search, upload_documents
    PIPELINE_STATUS["task8_pageindex"] = True
except ImportError:
    pass

try:
    from src.task10_generation import generate_with_citation as _gen
    generate_with_citation = _gen
    PIPELINE_STATUS["task10_generation"] = True
except ImportError:
    pass

try:
    from src.task9_retrieval_pipeline import retrieve as _ret
    retrieve = _ret
    PIPELINE_STATUS["task9_retrieval"] = True
except ImportError:
    pass


# =============================================================================
# CUSTOM CSS — Editorial Premium (Later-inspired)
# =============================================================================

st.markdown("""
<style>
    /* ==== Typography & color tokens ==== */
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --ink: #0e0e0e;
        --ink-soft: #2a2a2a;
        --ink-mid: #4a4a4a;
        --ink-mute: #8a8a8a;
        --line: #ebe7df;
        --line-soft: #f0ece4;
        --cream: #faf7f2;
        --cream-deep: #f3eee5;
        --accent: #ff5722;     /* signature warm coral (Later-style) */
        --accent-soft: #ffe7df;
        --gold: #b8865b;
        --good: #4f7942;
        --good-soft: #e8efe5;
        --bad: #b5483a;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--ink);
    }

    .stApp {
        background: var(--cream);
        background-image:
            radial-gradient(at 12% 8%, rgba(255, 87, 34, 0.04) 0%, transparent 35%),
            radial-gradient(at 92% 90%, rgba(184, 134, 91, 0.05) 0%, transparent 40%);
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Subtle scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--line); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--ink-mute); }

    /* ==== Sidebar ==== */
    [data-testid="stSidebar"] {
        background: var(--cream-deep) !important;
        border-right: 1px solid var(--line) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    /* Sidebar logo block */
    .sidebar-logo {
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.5rem;
    }
    .sidebar-logo .mark {
        font-family: 'Fraunces', serif;
        font-size: 2.4rem;
        font-weight: 600;
        color: var(--accent);
        line-height: 1;
        letter-spacing: -0.04em;
    }
    .sidebar-logo .name {
        font-family: 'Fraunces', serif;
        font-size: 1.05rem;
        font-weight: 500;
        color: var(--ink);
        margin-top: 0.5rem;
        letter-spacing: -0.01em;
    }
    .sidebar-logo .sub {
        font-size: 0.7rem;
        color: var(--ink-mute);
        margin-top: 0.35rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    .sidebar-section-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--ink-mute);
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .status-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.65rem 0.85rem;
        background: white;
        border: 1px solid var(--line);
        border-radius: 10px;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .status-row .label { color: var(--ink-soft); }
    .status-row .dot {
        width: 8px; height: 8px; border-radius: 50%;
        display: inline-block; margin-right: 0.5rem;
    }
    .status-row .dot.up { background: var(--good); box-shadow: 0 0 0 3px var(--good-soft); }
    .status-row .dot.down { background: var(--ink-mute); }
    .status-row .status-text {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .status-row .status-text.up { color: var(--good); }
    .status-row .status-text.down { color: var(--ink-mute); }

    /* Sidebar stButtons — flat ghost style */
    [data-testid="stSidebar"] .stButton > button {
        background: white;
        color: var(--ink);
        border: 1px solid var(--line);
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 500;
        text-align: left;
        padding: 0.65rem 0.85rem;
        transition: all 180ms ease;
        box-shadow: none;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--ink);
        background: var(--ink);
        color: var(--cream);
        transform: translateY(-1px);
    }

    /* Sidebar slider/checkbox */
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: var(--ink-soft) !important;
        font-size: 0.85rem;
    }

    /* ==== Main: hero ==== */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.25rem 0 0.5rem 0;
        border-bottom: 1px solid var(--line);
        margin-bottom: 3.5rem;
    }
    .top-nav .brand {
        font-family: 'Fraunces', serif;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: var(--ink);
    }
    .top-nav .brand .mark { color: var(--accent); }
    .top-nav .crumbs {
        font-size: 0.72rem;
        color: var(--ink-mute);
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }
    .top-nav .crumbs span { color: var(--ink); }

    .hero {
        padding: 2rem 0 4rem 0;
        max-width: 1100px;
    }
    .hero .eyebrow {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 1.5rem;
        padding: 0.4rem 0.85rem;
        background: var(--accent-soft);
        border-radius: 999px;
    }
    .hero h1 {
        font-family: 'Fraunces', serif;
        font-weight: 400;
        font-size: clamp(2.6rem, 6vw, 4.6rem);
        line-height: 0.98;
        letter-spacing: -0.04em;
        color: var(--ink);
        margin: 0 0 1.5rem 0;
    }
    .hero h1 em {
        font-style: italic;
        font-weight: 400;
        color: var(--accent);
    }
    .hero h1 .underline {
        position: relative;
        display: inline-block;
    }
    .hero h1 .underline::after {
        content: '';
        position: absolute;
        bottom: 0.08em;
        left: 0; right: 0;
        height: 4px;
        background: var(--accent);
        opacity: 0.85;
    }
    .hero p.lede {
        font-size: 1.15rem;
        line-height: 1.55;
        color: var(--ink-mid);
        max-width: 640px;
        font-weight: 400;
        margin: 1rem 0 2rem 0;
    }

    .hero-meta {
        display: flex;
        gap: 2.5rem;
        padding-top: 2rem;
        border-top: 1px solid var(--line);
        margin-top: 2.5rem;
    }
    .hero-meta .item .num {
        font-family: 'Fraunces', serif;
        font-size: 2rem;
        font-weight: 500;
        color: var(--ink);
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .hero-meta .item .lbl {
        font-size: 0.72rem;
        color: var(--ink-mute);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-top: 0.5rem;
    }

    /* ==== Bento grid ==== */
    .section-heading {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin: 4rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--line);
    }
    .section-heading h2 {
        font-family: 'Fraunces', serif;
        font-size: 1.75rem;
        font-weight: 500;
        letter-spacing: -0.02em;
        color: var(--ink);
        margin: 0;
    }
    .section-heading .micro {
        font-size: 0.72rem;
        color: var(--ink-mute);
        text-transform: uppercase;
        letter-spacing: 0.18em;
    }

    .bento {
        display: grid;
        grid-template-columns: repeat(12, 1fr);
        gap: 1rem;
    }
    .bento-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.5rem;
        transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
        position: relative;
        overflow: hidden;
    }
    .bento-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(14,14,14,0.06);
        border-color: var(--ink);
    }
    .bento-card.dark {
        background: var(--ink);
        color: var(--cream);
        border-color: var(--ink);
    }
    .bento-card.dark .lbl { color: rgba(250,247,242,0.6); }
    .bento-card.cream {
        background: var(--cream-deep);
        border-color: transparent;
    }
    .bento-card.accent {
        background: var(--accent);
        color: white;
        border-color: var(--accent);
    }
    .bento-card.accent .lbl { color: rgba(255,255,255,0.75); }

    .bento-card.span-3 { grid-column: span 3; }
    .bento-card.span-4 { grid-column: span 4; }
    .bento-card.span-5 { grid-column: span 5; }
    .bento-card.span-6 { grid-column: span 6; }
    .bento-card.span-7 { grid-column: span 7; }
    .bento-card.span-8 { grid-column: span 8; }
    .bento-card.span-12 { grid-column: span 12; }

    .bento-card .lbl {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--ink-mute);
        text-transform: uppercase;
        letter-spacing: 0.18em;
        margin-bottom: 0.85rem;
        display: block;
    }
    .bento-card .num {
        font-family: 'Fraunces', serif;
        font-size: 3rem;
        font-weight: 400;
        line-height: 1;
        letter-spacing: -0.04em;
    }
    .bento-card .num.med { font-size: 2.2rem; }
    .bento-card .num.small { font-size: 1.4rem; }
    .bento-card .desc {
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: var(--ink-mid);
        line-height: 1.5;
    }
    .bento-card.dark .desc { color: rgba(250,247,242,0.7); }
    .bento-card.accent .desc { color: rgba(255,255,255,0.85); }

    /* Pipeline mini chips */
    .pipeline-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 1rem;
    }
    .chip {
        font-size: 0.7rem;
        font-weight: 500;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: var(--cream-deep);
        color: var(--ink-mid);
        border: 1px solid var(--line);
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    .chip .dot {
        width: 6px; height: 6px; border-radius: 50%;
    }
    .chip .dot.up { background: var(--good); }
    .chip .dot.down { background: var(--ink-mute); }
    .chip.up { background: var(--good-soft); border-color: transparent; color: var(--good); }

    /* ==== Chat area ==== */
    .chat-shell {
        background: white;
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 2rem 2rem 0 2rem;
        margin-top: 2.5rem;
        box-shadow: 0 4px 30px rgba(14,14,14,0.03);
    }
    .chat-shell-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.75rem;
    }
    .chat-shell-header .title {
        font-family: 'Fraunces', serif;
        font-size: 1.4rem;
        font-weight: 500;
        letter-spacing: -0.02em;
    }
    .chat-shell-header .meta {
        font-size: 0.7rem;
        color: var(--ink-mute);
        text-transform: uppercase;
        letter-spacing: 0.18em;
    }

    .thread {
        max-width: 820px;
        margin: 0 auto;
        padding-bottom: 1.5rem;
    }

    .bubble-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.75rem;
        animation: bubbleIn 420ms ease both;
    }
    @keyframes bubbleIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .bubble-row.user { justify-content: flex-end; }
    .bubble-row.assistant { justify-content: flex-start; }

    .avatar-mini {
        flex-shrink: 0;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        font-family: 'Fraunces', serif;
        font-weight: 600;
    }
    .avatar-mini.user {
        background: var(--ink);
        color: var(--cream);
    }
    .avatar-mini.assistant {
        background: var(--accent);
        color: white;
    }

    .bubble {
        max-width: 80%;
        padding: 1rem 1.25rem;
        line-height: 1.65;
        font-size: 0.95rem;
        border-radius: 18px;
    }
    .bubble.user {
        background: var(--ink);
        color: var(--cream);
        border-bottom-right-radius: 6px;
    }
    .bubble.assistant {
        background: var(--cream-deep);
        color: var(--ink);
        border-bottom-left-radius: 6px;
    }

    .bubble-meta {
        font-size: 0.7rem;
        color: var(--ink-mute);
        margin-top: 0.5rem;
        padding: 0 0.6rem;
        letter-spacing: 0.05em;
    }
    .bubble-row.user .bubble-meta { text-align: right; }

    /* Source cards */
    .source-card {
        background: white;
        border: 1px solid var(--line);
        border-left: 3px solid var(--accent);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.6rem 0;
    }
    .source-card .meta {
        font-size: 0.7rem;
        color: var(--ink-mute);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.4rem;
    }
    .source-card .content {
        font-size: 0.85rem;
        color: var(--ink-mid);
        line-height: 1.55;
    }

    /* Expander styled to match */
    .streamlit-expander {
        border: 1px solid var(--line) !important;
        background: var(--cream-deep) !important;
        border-radius: 12px !important;
    }
    .streamlit-expander details summary {
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--ink-soft) !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: white;
        border-radius: 999px;
        border: 1px solid var(--line);
        box-shadow: 0 4px 20px rgba(14,14,14,0.06);
    }

    /* Footer */
    .footer {
        margin-top: 4rem;
        padding: 2rem 0 3rem 0;
        border-top: 1px solid var(--line);
        text-align: center;
        font-size: 0.78rem;
        color: var(--ink-mute);
    }
    .footer .accent { color: var(--accent); font-weight: 600; }

    /* Responsive tweaks */
    @media (max-width: 900px) {
        .bento-card.span-3, .bento-card.span-4,
        .bento-card.span-5, .bento-card.span-6,
        .bento-card.span-7, .bento-card.span-8 {
            grid-column: span 12;
        }
        .hero-meta { gap: 1.5rem; flex-wrap: wrap; }
        .chat-shell { padding: 1.25rem 1rem 0 1rem; }
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="mark">✦</div>
        <div class="name">IELTS Assistant</div>
        <div class="sub">Band 8.0+ • RAG Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    # System health
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">System Health</div>', unsafe_allow_html=True)

    chroma_ok = PIPELINE_STATUS["task4_chunking"]
    llm_ok = PIPELINE_STATUS["task10_generation"]

    st.markdown(f"""
    <div class="status-row">
        <span><span class="dot {'up' if chroma_ok else 'down'}"></span>ChromaDB</span>
        <span class="status-text {'up' if chroma_ok else 'down'}">{'Online' if chroma_ok else 'Offline'}</span>
    </div>
    <div class="status-row">
        <span><span class="dot {'up' if llm_ok else 'down'}"></span>LLM</span>
        <span class="status-text {'up' if llm_ok else 'down'}">{'Online' if llm_ok else 'Offline'}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Quick actions
    st.markdown('<div class="sidebar-section-label">Quick Actions</div>', unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True, key="clear_chat"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Xin chào! Mình là trợ lý IELTS Writing, sẵn sàng hỗ trợ bạn tra cứu Band Descriptors, chiến lược viết bài Band 8.0+, và phân tích tiêu chí chấm điểm. Hãy đặt câu hỏi để bắt đầu nhé.",
        }]
        st.rerun()

    st.markdown('<div class="sidebar-section-label" style="margin-top: 1.5rem;">Try These</div>', unsafe_allow_html=True)
    suggestions = [
        "Sự khác biệt Band 6.0 và Band 7.0?",
        "Làm sao để đạt Band 8.0?",
        "Coherence & Cohesion là gì?",
        "Lexical Resource Band 9.0?",
    ]
    for i, sug in enumerate(suggestions):
        if st.button(sug, use_container_width=True, key=f"sug_{i}"):
            st.session_state.pending_query = sug

    st.markdown('<div class="sidebar-section-label" style="margin-top: 1.5rem;">Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("References", 3, 10, 5)
    show_sources = st.checkbox("Show sources", value=True)

    st.markdown(f"""
    <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid var(--line);">
        <div style="font-family: 'Fraunces', serif; font-size: 1.6rem; font-weight: 500; color: var(--ink); letter-spacing: -0.03em;">
            {st.session_state.get('query_count', 0)}
        </div>
        <div style="font-size: 0.7rem; color: var(--ink-mute); text-transform: uppercase; letter-spacing: 0.18em; margin-top: 0.3rem;">
            Queries this session
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Xin chào! Mình là trợ lý IELTS Writing, sẵn sàng hỗ trợ bạn tra cứu Band Descriptors, chiến lược viết bài Band 8.0+, và phân tích tiêu chí chấm điểm. Hãy đặt câu hỏi để bắt đầu nhé.",
    }]

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# =============================================================================
# HERO + META
# =============================================================================

st.markdown("""
<div class="top-nav">
    <div class="brand"><span class="mark">✦</span> IELTS Assistant</div>
    <div class="crumbs">Home <span>/</span> Writing Chat</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
    <span class="eyebrow">RAG-Powered • Band 8.0+ Strategy</span>
    <h1>Write essays that<br/>command <em>attention.</em></h1>
    <p class="lede">
        Tra cứu Band Descriptors, nghiên cứu chiến lược đạt điểm cao, và nhận phản hồi từ
        pipeline hybrid search kết hợp semantic + lexical retrieval cùng LLM sinh
        citation. Được xây dựng cho người viết nghiêm túc.
    </p>
    <div class="hero-meta">
        <div class="item">
            <div class="num">{DOC_STATS['total_docs']}</div>
            <div class="lbl">Source documents</div>
        </div>
        <div class="item">
            <div class="num">{DOC_STATS['total_chunks']}</div>
            <div class="lbl">Indexed chunks</div>
        </div>
        <div class="item">
            <div class="num">7</div>
            <div class="lbl">Pipeline stages</div>
        </div>
        <div class="item">
            <div class="num">Hybrid</div>
            <div class="lbl">Semantic + BM25</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# BENTO: Pipeline overview
# =============================================================================

pipeline_items = [
    ("T4 Chunking", PIPELINE_STATUS["task4_chunking"]),
    ("T5 Semantic", PIPELINE_STATUS["task5_semantic"]),
    ("T6 BM25", PIPELINE_STATUS["task6_lexical"]),
    ("T7 Rerank", PIPELINE_STATUS["task7_reranking"]),
    ("T8 PageIndex", PIPELINE_STATUS["task8_pageindex"]),
    ("T9 Pipeline", PIPELINE_STATUS["task9_retrieval"]),
    ("T10 Generation", PIPELINE_STATUS["task10_generation"]),
]

live_count = sum(1 for _, ok in pipeline_items if ok)
pipeline_chips_html = ""
for label, ok in pipeline_items:
    cls = "up" if ok else ""
    dot_cls = "up" if ok else "down"
    pipeline_chips_html += f'<span class="chip {cls}"><span class="dot {dot_cls}"></span>{label}</span>'

st.markdown("""
<div class="section-heading">
    <h2>The RAG pipeline, end-to-end.</h2>
    <span class="micro">Live status</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="bento">
    <div class="bento-card dark span-7">
        <span class="lbl">Pipeline overview</span>
        <div class="num med">7 <span style="font-size: 1rem; color: rgba(250,247,242,0.5);">/ 7</span> stages live</div>
        <div class="desc">Từ chunking tài liệu, semantic search, lexical BM25, rerank bằng RRF, tới generation với citation — tất cả chạy trên ChromaDB local kết hợp LLM cloud.</div>
        <div class="pipeline-chips">__PIPELINE_CHIPS__</div>
    </div>
    <div class="bento-card cream span-5">
        <span class="lbl">Stack</span>
        <div class="num small" style="font-family: 'Inter', sans-serif; font-weight: 500; line-height: 1.4;">
            ChromaDB<br/>
            Sentence-Transformers<br/>
            BM25 &middot; RRF<br/>
            GPT-4o
        </div>
    </div>
    <div class="bento-card span-4">
        <span class="lbl">Semantic</span>
        <div class="num small">Dense</div>
        <div class="desc">Embedding cosine similarity, HyDE-augmented queries.</div>
    </div>
    <div class="bento-card span-4">
        <span class="lbl">Lexical</span>
        <div class="num small">BM25</div>
        <div class="desc">Sparse retrieval bắt trúng exact-match terms trong tài liệu.</div>
    </div>
    <div class="bento-card accent span-4">
        <span class="lbl">Hybrid</span>
        <div class="num small" style="color: white;">RRF Fusion</div>
        <div class="desc">Reciprocal rank fusion để hợp nhất hai retrieval streams trước khi rerank.</div>
    </div>
</div>
""".replace("__PIPELINE_CHIPS__", pipeline_chips_html), unsafe_allow_html=True)


# =============================================================================
# CHAT SECTION
# =============================================================================

st.markdown("""
<div class="section-heading">
    <h2>Ask the assistant.</h2>
    <span class="micro">Streaming context-aware answers</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="chat-shell">
    <div class="chat-shell-header">
        <span class="title">Writing assistant</span>
        <span class="meta">Hybrid search · RRF · Citations</span>
    </div>
    <div class="thread">
""", unsafe_allow_html=True)

# Render history
for msg in st.session_state.messages:
    timestamp = datetime.now().strftime("%H:%M")
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="bubble-row user">
            <div>
                <div class="bubble user">{msg['content']}</div>
                <div class="bubble-meta">{timestamp}</div>
            </div>
            <div class="avatar-mini user">B</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bubble-row assistant">
            <div class="avatar-mini assistant">✦</div>
            <div>
                <div class="bubble assistant">{msg['content']}</div>
                <div class="bubble-meta">{timestamp} · retrieved from corpus</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if show_sources and msg.get("sources"):
            with st.expander(f"{len(msg['sources'])} sources referenced", expanded=False):
                for src in msg["sources"]:
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "Unknown")
                    content = src.get("content", "")
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="meta">{source_name}</div>
                        <div class="content">{content[:280]}{"…" if len(content) > 280 else ""}</div>
                    </div>
                    """, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)


# =============================================================================
# CHAT HANDLER
# =============================================================================

def add_assistant_message(response: str, sources: list):
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })


# =============================================================================
# INPUT
# =============================================================================

user_input = st.chat_input("Ask about IELTS Writing strategy…")

# handle pending suggestion
if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

if user_input:
    st.session_state.query_count += 1
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        if generate_with_citation is not None and PIPELINE_STATUS["task9_retrieval"]:
            with st.spinner("Tra cứu corpus và sinh phản hồi…"):
                result = generate_with_citation(user_input, top_k=top_k)
                response = result.get("answer", "Xin lỗi, có lỗi xảy ra.")
                sources = result.get("sources", [])
        else:
            response = "Hệ thống đang trong quá trình kích hoạt. Vui lòng thử lại sau."
            sources = []

        add_assistant_message(response, sources)
    except Exception as e:
        add_assistant_message(f"Đã xảy ra lỗi: {e}", [])

    st.rerun()


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div class="footer">
    <span class="accent">✦</span> IELTS Writing Assistant · RAG Pipeline v3
    &nbsp;·&nbsp; Hybrid Search <span style="color: var(--ink-mute;">·</span> RRF <span style="color: var(--ink-mute;">·</span> Citations
</div>
""", unsafe_allow_html=True)
