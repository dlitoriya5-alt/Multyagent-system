import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import scrape_url, web_search

load_dotenv()

# Support both GEMINI_API_KEY and GOOGLE_API_KEY
api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip().strip('"\'')
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key

model_name = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key if api_key else None,
    temperature=0.2,
)


def extract_text_content(content) -> str:
    """Safely extracts clean string text from LangChain message content (handling list of blocks/signatures)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
            elif hasattr(part, "text"):
                parts.append(str(getattr(part, "text")))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content) if content is not None else ""


def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are an expert research search agent. "
            "Use the web_search tool to gather reliable, recent, and factual information on the requested topic. "
            "Always include source URLs and key snippets in your response."
        ),
    )


def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are an expert web reader agent. "
            "Review the provided search findings, pick the single most authoritative and relevant URL, "
            "and call the scrape_url tool to extract comprehensive details for deep analysis."
        ),
    )


# Writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research analyst and technical writer. "
        "Create clear, structured, highly professional, and insightful reports.",
    ),
    (
        "human",
        """Write a comprehensive, publication-grade research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report with the following markdown sections:
# Executive Summary
## Introduction
## Key Findings (detailed, minimum 3 well-explained points with supporting facts)
## Market / Industry Implications
## Conclusion & Strategic Outlook
## Sources & References (list all cited URLs found in the research)

Formatting guidelines:
- Use clean Markdown with headers, bullet points, and bold text for key terms.
- Be objective, detailed, factual, and articulate.""",
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# Critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a rigorous, objective academic and editorial research critic. "
        "Evaluate reports strictly based on depth, factual rigor, structure, clarity, and source validity.",
    ),
    (
        "human",
        """Review the research report below and provide a structured, constructive evaluation.

Report to Review:
{report}

Respond strictly in this Markdown format:

### 📊 Overall Score: X/10

### 🌟 Key Strengths:
- ...
- ...
- ...

### ⚠️ Areas for Improvement:
- ...
- ...

### 🎯 Actionable Recommendations:
- ...
- ...

### 💡 One-Line Verdict:
...""",
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
