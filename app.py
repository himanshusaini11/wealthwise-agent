import streamlit as st
import pandas as pd
from src.graph import process_query

st.set_page_config(page_title="WealthWise", page_icon="💰")

st.title("WealthWise: Your AI Financial Analyst")

# --- SIDEBAR ---
st.sidebar.title("Model Settings")

provider = st.sidebar.radio(
    "Choose model",
    options=["claude-haiku", "claude-sonnet", "gemini", "groq"],
    format_func=lambda x: {
        "claude-haiku": "Claude Haiku 3.5 (Fast)",
        "claude-sonnet": "Claude Sonnet 4.5 (Smart)",
        "gemini": "Gemini 2.0 Flash",
        "groq": "Llama 3.3 70B (Groq)",
    }[x],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Data Preview")
try:
    df = pd.read_csv("data/transactions.csv")
    st.sidebar.dataframe(df.head())
except FileNotFoundError:
    st.sidebar.warning("data/transactions.csv not found. Please run scripts/generate_data.py first.")

st.sidebar.divider()
st.sidebar.subheader("Session Stats")

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

st.sidebar.metric("Queries", st.session_state.query_count)
st.sidebar.metric("Input tokens", f"{st.session_state.total_input_tokens:,}")
st.sidebar.metric("Output tokens", f"{st.session_state.total_output_tokens:,}")

if "last_usage" in st.session_state:
    usage = st.session_state.last_usage
    pct = usage["total_tokens"] / usage["context_window_limit"]
    st.sidebar.divider()
    st.sidebar.caption(f"Context window: {usage['model_name']}")
    st.sidebar.progress(min(pct, 1.0))
    st.sidebar.caption(
        f"{usage['total_tokens']:,} / {usage['context_window_limit']:,} tokens "
        f"({pct*100:.1f}%)"
    )

# --- MAIN CHAT AREA ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.write(message["content"])
        else:
            st.write(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        response_text, usage = process_query(
            prompt,
            provider=provider,
            thread_id=st.session_state.thread_id,
        )
    except RuntimeError as e:
        response_text = str(e)
        usage = {"input_tokens": 0, "output_tokens": 0,
                 "total_tokens": 0, "model_name": provider,
                 "context_window_limit": 200000}
    except Exception:
        response_text = "Something went wrong. Please try again."
        usage = {"input_tokens": 0, "output_tokens": 0,
                 "total_tokens": 0, "model_name": provider,
                 "context_window_limit": 200000}

    st.session_state.total_input_tokens += usage["input_tokens"]
    st.session_state.total_output_tokens += usage["output_tokens"]
    st.session_state.query_count += 1
    st.session_state.last_usage = usage

    with st.chat_message("assistant"):
        st.write(response_text)
        if usage["input_tokens"] > 0:
            st.caption(
                f"↑ {usage['input_tokens']:,} in · "
                f"↓ {usage['output_tokens']:,} out · "
                f"{usage['model_name']}"
            )

    st.session_state.messages.append({"role": "assistant", "content": response_text})
