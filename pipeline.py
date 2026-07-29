from agents import build_reader_agent, build_search_agent , writer_chain,critic_chain

def run_reasearch_pipeline(topic : str) ->dict:

    state = {}

    #search agent working
    print("\n"+" ="*50)
    print("step 1 - search agent is working .....")
    print("="*50)


    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"find recent , reliable and detailed information about : {topic}")]
    })
    state["search_result"] = search_result["messages"][-1].content

    print("\n search result ",state['search_result'])



    print("\n"+" ="*50)
    print("step 2 - search agent is working .....")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
                  f"based on the following results about '{topic}',"
                  f"pick the most relavent URl and scrape it for deeper content.\n\n"
                  f"search result : \n{state['search_result'][:800]}")]
    })

    state["scraped_content"] =reader_result["messages"][-1].content

    print("\nscraped content\n",state["scraped_content"])


    print("\n"+" ="*50)
    print("step 3 - writer is drafting the report  .....")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_result']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined 
    })

    print("\n final report \n",state["report"])



    print("\n"+" ="*50)
    print("step 3 - critic is reviewing the report  .....")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\n critic report \n ",state["feedback"])

    return state

if __name__ == "__main__":
    topic = input("\n enater a research topic:")
    run_reasearch_pipeline(topic)
