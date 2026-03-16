import logging
import os
import time
from datetime import date
from typing import Annotated

from groq import BadRequestError as GroqBadRequestError
from groq import RateLimitError as GroqRateLimitError
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic import APIStatusError as AnthropicAPIStatusError
from typing_extensions import TypedDict

from .config import get_settings
from .logger import setup_logging
from .tools import _build_python_analyst, predict_spending_trend

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()

CONTEXT_WINDOW_LIMITS = {
    "claude-haiku": 200000,
    "claude-sonnet": 200000,
    "gemini": 1000000,
    "groq": 128000,
}

CHECKPOINT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "checkpoints.db"
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _should_continue(state: AgentState) -> str:
    """
    Route to END if:
    - Last message is an AIMessage with no tool calls (LLM finished)
    - Last ToolMessage contains PREDICTION COMPLETE
    Otherwise continue the loop.
    """
    messages = state["messages"]
    last = messages[-1]

    # LLM decided to stop (no tool calls)
    if isinstance(last, AIMessage):
        if not getattr(last, "tool_calls", None):
            return END
        return "tools"  # has tool calls — execute them

    # Tool returned a prediction — stop immediately
    if isinstance(last, ToolMessage):
        if "PREDICTION COMPLETE" in (last.content or ""):
            logger.info("PREDICTION COMPLETE signal received — routing to END")
            return END

    return "agent"


def get_llm(provider: str | None = None):
    """
    Returns an LLM instance for the given provider.
    Falls back to the alternate provider if the primary fails.
    """
    target = (provider or settings.model_provider).lower()

    try:
        if target == "claude-haiku":
            logger.info("Initialising LLM: claude-haiku-3-5-20251001 via Anthropic")
            return ChatAnthropic(
                model="claude-haiku-3-5-20251001",
                temperature=0,
                max_retries=2,
                api_key=settings.anthropic_api_key,
            )
        elif target == "claude-sonnet":
            logger.info("Initialising LLM: claude-sonnet-4-5-20251014 via Anthropic")
            return ChatAnthropic(
                model="claude-sonnet-4-5-20251014",
                temperature=0,
                max_retries=2,
                api_key=settings.anthropic_api_key,
            )
        elif target == "gemini":
            logger.info("Initialising LLM: gemini-2.0-flash via Google")
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                max_retries=2,
                google_api_key=settings.google_api_key,
            )
        else:  # groq
            logger.info("Initialising LLM: llama-3.3-70b-versatile via Groq")
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_retries=2,
                api_key=settings.groq_api_key,
            )
    except Exception as e:
        fallback = "claude-haiku" if target != "claude-haiku" else "groq"
        logger.warning(
            "Primary LLM failed, attempting fallback",
            extra={"primary": target, "fallback": fallback, "error": str(e)},
        )
        return get_llm(provider=fallback)


def _build_agent(provider: str):
    llm = get_llm(provider)
    tools = [predict_spending_trend, _build_python_analyst()]
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent_node(state: AgentState):
        logger.debug("Agent node called, message count: %d", len(state["messages"]))
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_conditional_edges(
        "tools",
        _should_continue,
        {"agent": "agent", END: END},
    )

    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    logger.info("Agent built", extra={"provider": provider})
    return graph.compile(checkpointer=checkpointer)


def process_query(query: str, provider: str = "claude-haiku", thread_id: str = "default") -> tuple[str, dict]:
    """
    Invokes the agent and returns (response_text, usage_metadata).
    Raises RuntimeError on unrecoverable failure.
    """
    logger.info("process_query called", extra={"query": query[:120], "provider": provider, "thread_id": thread_id})
    start = time.perf_counter()

    today = date.today()
    today_str = today.strftime("%B %d, %Y")
    last_month_num = today.month - 1 if today.month > 1 else 12
    last_month_year = today.year if today.month > 1 else today.year - 1

    system_instruction = SystemMessage(content=f"""You are a WealthWise financial advisor.
    Today's date is {today_str}.

    INSTRUCTIONS:
    1. DATA: Use 'python_analyst' for questions about past transactions.
       DataFrame 'df' has columns: Date (datetime64), Category (title case),
       Amount (negative=expense, positive=income), Description.

       Date filtering rules — always use these exact patterns:
       - Current month: df[df['Date'].dt.month == {today.month}]
       - Last month is month {last_month_num} of year {last_month_year}.
         Use: mask = (df['Date'].dt.month == {last_month_num}) & (df['Date'].dt.year == {last_month_year})
         Then: df[mask & (df['Amount'] < 0)]['Amount'].abs().sum()
       - Specific category: df[df['Category'] == 'Food']  (always title case)
       - Expenses only: df[df['Amount'] < 0]
       - Spend amount: df[mask]['Amount'].abs().sum()
    2. FORECASTING: Use 'predict_spending_trend' for future spending questions.
       Pass the number of days as a digit string e.g. "14", never a written number.
    3. STOPPING: Once you receive a tool result starting with "PREDICTION COMPLETE",
       stop calling tools and report the result directly to the user.
    4. DATE QUESTIONS: You know today's date. Answer directly without using any tool.
    """)

    inputs = {"messages": [system_instruction, ("user", query)]}

    config = {
        "recursion_limit": 10,
        "configurable": {"thread_id": thread_id},
    }

    MAX_RETRIES = 3

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug("Agent invoking with query: %s", query)
            agent = _build_agent(provider)
            result = agent.invoke(inputs, config=config)
            break  # success — exit retry loop
        except GroqBadRequestError as e:
            if "tool_use_failed" in str(e):
                logger.warning(
                    "Groq tool_use_failed on attempt %d/%d, retrying",
                    attempt, MAX_RETRIES,
                    extra={"error": str(e)[:200]},
                )
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        "The model failed to call the tool correctly after "
                        f"{MAX_RETRIES} attempts. Please rephrase your question."
                    ) from e
                continue
            raise RuntimeError(
                "The agent encountered an unexpected error. Please try again."
            ) from e
        except GroqRateLimitError as e:
            raise RuntimeError(
                "Groq rate limit reached. Please select a different model from the sidebar."
            ) from e
        except AnthropicRateLimitError as e:
            raise RuntimeError(
                "Claude API rate limit reached. Please select a different model or wait a moment."
            ) from e
        except AnthropicAPIStatusError as e:
            if e.status_code in (429, 529):
                raise RuntimeError(
                    "Claude API is overloaded. Please select a different model from the sidebar."
                ) from e
            raise RuntimeError(
                "The agent encountered an unexpected error. Please try again."
            ) from e
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "rate limit" in error_str or "429" in error_str:
                raise RuntimeError(
                    f"The selected model's quota is exhausted. "
                    f"Please select a different model from the sidebar."
                ) from e
            logger.exception("Agent invocation failed — full traceback:")
            raise RuntimeError(
                "The agent encountered an unexpected error. Please try again."
            ) from e

    elapsed = time.perf_counter() - start
    logger.info("process_query complete", extra={"elapsed_s": round(elapsed, 3)})

    response_text = _extract_response(result)
    usage_metadata = _extract_usage(result, provider)
    return response_text, usage_metadata


def _extract_response(result: dict | list | str) -> str:
    """Safely extracts a plain string from the agent result."""
    try:
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                return last.content if hasattr(last, "content") else str(last)
            return result.get("output", str(result))
        if isinstance(result, list):
            return " ".join(str(m) for m in result)
        return str(result)
    except Exception as e:
        logger.warning("Response extraction failed", extra={"error": str(e)})
        return "I was unable to process that response. Please try again."


def _extract_usage(result: dict, provider: str) -> dict:
    """Extracts token usage metadata from the agent result."""
    input_tokens = 0
    output_tokens = 0
    try:
        last_message = result["messages"][-1]
        usage = getattr(last_message, "usage_metadata", None)
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
    except Exception as e:
        logger.warning("Usage extraction failed", extra={"error": str(e)})

    model_names = {
        "claude-haiku": "claude-haiku-3-5-20251001",
        "claude-sonnet": "claude-sonnet-4-5-20251014",
        "gemini": "gemini-2.0-flash",
        "groq": "llama-3.3-70b-versatile",
    }
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "model_name": model_names.get(provider, provider),
        "context_window_limit": CONTEXT_WINDOW_LIMITS.get(provider, 200000),
    }
