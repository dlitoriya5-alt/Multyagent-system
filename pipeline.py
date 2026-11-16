from agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain,
    extract_text_content,
    writer_chain,
)


def run_research_pipeline(topic: str) -> dict:
    state = {}

    # Step 1: Search Agent
    print("\n" + "=" * 50)
    print("Step 1 - Search Agent is gathering information...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable, and detailed information about: {topic}")]
    })
    state["search_result"] = extract_text_content(search_result["messages"][-1].content)
    print("\n[Search Results Summary]:\n", state["search_result"][:500], "...\n")

    # Step 2: Reader Agent
    print("\n" + "=" * 50)
    print("Step 2 - Reader Agent is scraping the best source for deeper insights...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [(
            "user",
            f"Based on the following search results about '{topic}', "
            f"pick the single most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_result']}"
        )]
    })
    state["scraped_content"] = extract_text_content(reader_result["messages"][-1].content)
    print("\n[Scraped Content Preview]:\n", state["scraped_content"][:500], "...\n")

    # Step 3: Writer Chain
    print("\n" + "=" * 50)
    print("Step 3 - Writer is drafting the comprehensive research report...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_result']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    state["report"] = extract_text_content(
        writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        })
    )
    print("\n[Final Report Preview]:\n", state["report"][:500], "...\n")

    # Step 4: Critic Chain
    print("\n" + "=" * 50)
    print("Step 4 - Critic is reviewing and scoring the report...")
    print("=" * 50)

    state["feedback"] = extract_text_content(
        critic_chain.invoke({
            "report": state["report"],
        })
    )
    print("\n[Critic Review]:\n", state["feedback"])

    return state


# Backward compatibility alias
run_reasearch_pipeline = run_research_pipeline

if __name__ == "__main__":
    user_topic = input("\nEnter a research topic: ").strip()
    if user_topic:
        run_research_pipeline(user_topic)
    else:
        print("No topic provided. Exiting.")
