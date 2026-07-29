

import streamlit as st
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Multi-Agent Research System")
st.caption("Search agent → Reader agent → Writer → Critic")

# --- keep results across reruns ---
if "state" not in st.session_state:
    st.session_state.state = None

with st.sidebar:
    st.header("New Research")
    topic = st.text_input("Research topic", placeholder="e.g. Latest advances in solid-state batteries")
    run_clicked = st.button("🚀 Run pipeline", type="primary", use_container_width=True)
    st.divider()
    st.caption("Each step runs live below. This can take a minute or two depending on the topic.")


def run_pipeline_with_ui(topic: str) -> dict:
    """Same logic as pipeline.run_reasearch_pipeline, but streamed into the UI."""
    state = {}

    with st.status("Step 1 · Search agent is working...", expanded=True) as status:
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"find recent, reliable and detailed information about: {topic}")]
        })
        state["search_result"] = search_result["messages"][-1].content
        st.markdown(state["search_result"])
        status.update(label="Step 1 · Search complete ✅", state="complete")

    with st.status("Step 2 · Reader agent is scraping the best source...", expanded=True) as status:
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [("user",
                          f"based on the following results about '{topic}', "
                          f"pick the most relevant URL and scrape it for deeper content.\n\n"
                          f"search result:\n{state['search_result'][:800]}")]
        })
        state["scraped_content"] = reader_result["messages"][-1].content
        st.markdown(state["scraped_content"])
        status.update(label="Step 2 · Scraping complete ✅", state="complete")

    with st.status("Step 3 · Writer is drafting the report...", expanded=True) as status:
        research_combined = (
            f"SEARCH RESULTS:\n{state['search_result']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
        )
        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        })
        st.markdown(state["report"])
        status.update(label="Step 3 · Draft ready ✅", state="complete")

    with st.status("Step 4 · Critic is reviewing the report...", expanded=True) as status:
        state["feedback"] = critic_chain.invoke({
            "report": state["report"]
        })
        st.markdown(state["feedback"])
        status.update(label="Step 4 · Review complete ✅", state="complete")

    return state


if run_clicked:
    if not topic or not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.state = run_pipeline_with_ui(topic.strip())
        st.session_state.topic = topic.strip()

# --- final results, always visible after a run ---
if st.session_state.state:
    st.divider()
    st.subheader(f"📄 Final Report — {st.session_state.get('topic', '')}")

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["Report", "Critic feedback", "Raw search results", "Scraped content"]
    )

    with tab_report:
        st.markdown(st.session_state.state["report"])
        st.download_button(
            "⬇️ Download report (.md)",
            data=st.session_state.state["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(st.session_state.state["feedback"])

    with tab_search:
        st.markdown(st.session_state.state["search_result"])

    with tab_scraped:
        st.markdown(st.session_state.state["scraped_content"])
else:
    st.info("Enter a topic in the sidebar and click **Run pipeline** to get started.")