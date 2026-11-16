import os
import re
import time
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain,
    extract_text_content,
    llm,
    writer_chain,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="NexusAgent · Multi-Agent Deep Research",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# High-End Design System (CSS)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient header banner */
    .hero-banner {
        background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.15) 50%, rgba(15, 23, 42, 0) 100%),
                    linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.02rem;
        margin-bottom: 16px;
    }

    /* Pipeline workflow steps pill bar */
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 14px;
    }

    .step-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .step-card:hover {
        transform: translateY(-2px);
        border-color: rgba(129, 140, 248, 0.5);
    }

    .step-icon {
        font-size: 1.4rem;
        background: rgba(99, 102, 241, 0.15);
        padding: 8px;
        border-radius: 8px;
    }

    .step-title {
        font-weight: 700;
        font-size: 0.88rem;
        color: #f1f5f9;
    }

    .step-desc {
        font-size: 0.74rem;
        color: #94a3b8;
    }

    /* Status cards & metrics */
    .metric-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .badge-model {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar Configuration & Diagnostics
# ---------------------------------------------------------
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
tavily_key = os.getenv("TAVILY_API_KEY") or ""

with st.sidebar:
    st.image(
        "https://api.iconify.design/heroicons:cpu-chip-solid.svg?color=%23818cf8&width=48&height=48",
        width=48,
    )
    st.title("NexusAgent")
    st.caption("Autonomous Collaborative Intelligence")
    st.divider()

    # System Status Section
    st.subheader("System Status")
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if gemini_key:
            st.markdown('<span class="metric-badge badge-success">✓ Gemini Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="metric-badge badge-warning">⚠ No Gemini Key</span>', unsafe_allow_html=True)
    with col_k2:
        if tavily_key:
            st.markdown('<span class="metric-badge badge-success">✓ Tavily Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="metric-badge badge-warning">⚠ No Tavily Key</span>', unsafe_allow_html=True)

    current_model = getattr(llm, "model", "gemini-3.5-flash-lite")
    st.markdown(f'<div style="margin-top: 10px;"><span class="metric-badge badge-model">⚡ Engine: {current_model}</span></div>', unsafe_allow_html=True)
    st.divider()

    # Research Input
    st.subheader("Initiate Research")
    
    # Topic Preset Buttons
    st.caption("Quick Select Topic Ideas:")
    preset_topics = [
        "Quantum computing breakthroughs and practical timeline",
        "Latest advances in solid-state EV battery commercialization",
        "Autonomous multi-agent architectures in production",
        "CRISPR gene therapies and recent regulatory approvals",
    ]
    
    selected_preset = None
    for p in preset_topics:
        if st.button(p, key=f"btn_{p[:15]}", use_container_width=True):
            st.session_state["topic_input"] = p

    # Research topic text input
    default_text = st.session_state.get("topic_input", "")
    topic = st.text_area(
        "Enter Research Topic or Query:",
        value=default_text,
        placeholder="e.g., Deep dive into next-generation silicon photonics for AI accelerators...",
        height=100,
    )

    run_clicked = st.button("🚀 Run Multi-Agent Pipeline", type="primary", use_container_width=True)

    if st.session_state.get("state"):
        if st.button("🧹 Clear & Start Fresh", use_container_width=True):
            st.session_state.state = None
            st.session_state.topic = None
            st.rerun()

    st.divider()
    st.caption(f"Powered by Google Gemini ({current_model}) & LangGraph Agent Framework.")

# ---------------------------------------------------------
# Hero Banner & Architecture Display
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">Multi-Agent Autonomous Research System</div>
        <div class="hero-subtitle">
            Harnessing 4 specialized Gemini AI agents working sequentially: Real-time search discovery, deep web reading, synthesis drafting, and rigorous peer review.
        </div>
        <div class="pipeline-grid">
            <div class="step-card">
                <div class="step-icon">🔍</div>
                <div>
                    <div class="step-title">1. Search Agent</div>
                    <div class="step-desc">Tavily live web discovery</div>
                </div>
            </div>
            <div class="step-card">
                <div class="step-icon">📖</div>
                <div>
                    <div class="step-title">2. Reader Agent</div>
                    <div class="step-desc">Deep page scraping</div>
                </div>
            </div>
            <div class="step-card">
                <div class="step-icon">✍️</div>
                <div>
                    <div class="step-title">3. Writer Agent</div>
                    <div class="step-desc">Executive report synthesis</div>
                </div>
            </div>
            <div class="step-card">
                <div class="step-icon">⚖️</div>
                <div>
                    <div class="step-title">4. Critic Agent</div>
                    <div class="step-desc">Strict audit & scoring</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "state" not in st.session_state:
    st.session_state.state = None
if "topic" not in st.session_state:
    st.session_state.topic = ""


# ---------------------------------------------------------
# Pipeline Execution Flow
# ---------------------------------------------------------
def run_pipeline_with_ui(target_topic: str) -> dict:
    state = {}
    start_time = time.time()

    # Step 1: Search Agent
    with st.status("🔍 Step 1: Search Agent is discovering real-time sources...", expanded=True) as status_1:
        st.write("Target query:", target_topic)
        search_agent = build_search_agent()
        search_res = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {target_topic}")]
        })
        state["search_result"] = extract_text_content(search_res["messages"][-1].content)
        st.markdown(state["search_result"])
        status_1.update(label="✅ Step 1: Search Discovery Complete", state="complete")

    # Step 2: Reader Agent
    with st.status("📖 Step 2: Reader Agent is analyzing & scraping the primary source...", expanded=True) as status_2:
        reader_agent = build_reader_agent()
        search_snippet = state["search_result"]
        reader_res = reader_agent.invoke({
            "messages": [(
                "user",
                f"Based on the following research findings about '{target_topic}', "
                f"select the single most authoritative URL and scrape it for deep content.\n\n"
                f"Search Findings:\n{search_snippet}"
            )]
        })
        state["scraped_content"] = extract_text_content(reader_res["messages"][-1].content)
        st.markdown(state["scraped_content"][:1500] + ("..." if len(state["scraped_content"]) > 1500 else ""))
        status_2.update(label="✅ Step 2: Deep Reading & Scraping Complete", state="complete")

    # Step 3: Writer Chain
    with st.status("✍️ Step 3: Writer Agent is synthesizing comprehensive report...", expanded=True) as status_3:
        research_combined = (
            f"SEARCH RESULTS:\n{state['search_result']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
        )
        report_raw = writer_chain.invoke({
            "topic": target_topic,
            "research": research_combined,
        })
        state["report"] = extract_text_content(report_raw)
        st.markdown(state["report"][:1200] + "...")
        status_3.update(label="✅ Step 3: Synthesis Report Ready", state="complete")

    # Step 4: Critic Chain
    with st.status("⚖️ Step 4: Critic Agent is evaluating rigor, clarity, and scoring...", expanded=True) as status_4:
        feedback_raw = critic_chain.invoke({
            "report": state["report"]
        })
        state["feedback"] = extract_text_content(feedback_raw)
        st.markdown(state["feedback"])
        status_4.update(label="✅ Step 4: Peer Review & Audit Complete", state="complete")

    state["elapsed_time"] = round(time.time() - start_time, 1)
    return state


# Handle Execution Trigger
if run_clicked:
    clean_topic = topic.strip()
    if not clean_topic:
        st.warning("⚠️ Please enter a research topic or choose a preset idea from the sidebar.")
    elif not gemini_key:
        st.error("❌ Gemini API key not found! Please check your `.env` file for GOOGLE_API_KEY or GEMINI_API_KEY.")
    else:
        with st.spinner("Collaborative multi-agent swarm in progress..."):
            try:
                st.session_state.state = run_pipeline_with_ui(clean_topic)
                st.session_state.topic = clean_topic
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")

# ---------------------------------------------------------
# Results Presentation Display
# ---------------------------------------------------------
if st.session_state.state:
    res = st.session_state.state
    st.divider()

    # Metrics header
    score_match = re.search(r"(\d+(\.\d+)?)\s*/\s*10", res.get("feedback", ""))
    score_display = f"{score_match.group(1)} / 10" if score_match else "Evaluated"
    word_count = len(res.get("report", "").split())
    elapsed = res.get("elapsed_time", "N/A")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Critic Audit Score", score_display)
    with col_m2:
        st.metric("Report Word Count", f"{word_count:,} words")
    with col_m3:
        st.metric("Total Execution Time", f"{elapsed}s")
    with col_m4:
        st.metric("Agents Deployed", "4 Autonomous Agents")

    st.subheader(f"📑 Intelligence Dossier: {st.session_state.get('topic', '')}")

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs([
        "📄 Final Research Dossier",
        "⚖️ Rigorous Peer Review",
        "🔍 Search Evidence & URLs",
        "📖 Primary Source Scraped Data",
    ])

    with tab_report:
        st.markdown(res.get("report", "No report generated."))
        st.divider()
        col_d1, col_d2 = st.columns([1, 4])
        with col_d1:
            st.download_button(
                "⬇️ Download Markdown (.md)",
                data=res.get("report", ""),
                file_name=f"research_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with tab_feedback:
        st.markdown(res.get("feedback", "No feedback available."))

    with tab_search:
        st.info("Direct discoveries from Tavily Web Search agent:")
        st.markdown(res.get("search_result", "No search results."))

    with tab_scraped:
        st.info("In-depth content extracted by the Reader agent:")
        st.markdown(res.get("scraped_content", "No scraped content."))

else:
    # Helpful empty state
    st.info("💡 **Ready to begin**: Enter any inquiry or topic in the sidebar and click **🚀 Run Multi-Agent Pipeline**.")