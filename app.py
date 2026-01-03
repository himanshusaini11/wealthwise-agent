import streamlit as st
import pandas as pd
from src.graph import process_query

st.set_page_config(page_title="WealthWise", page_icon="💰")

st.title("WealthWise: Your AI Financial Analyst")

# Sidebar: Show a preview of the transactions.csv dataframe
st.sidebar.header("Data Preview")
try:
    df = pd.read_csv("data/transactions.csv")
    st.sidebar.dataframe(df.head())
except FileNotFoundError:
    st.sidebar.warning("data/transactions.csv not found. Please run scripts/generate_data.py first.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Interface: React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 4. Integration: Call process_query
    result = process_query(prompt)

    # --- ROBUST RESPONSE EXTRACTION LOGIC ---
    response_text = ""
    
    # Case 1: It's a Dictionary (The full Graph State)
    if isinstance(result, dict) and "messages" in result:
        final_content = result["messages"][-1].content
    else:
        # Case 2: It's already the content (String or List)
        final_content = result

    # Handle Gemini's "List of Blocks" Quirk
    if isinstance(final_content, list):
        # Join all text blocks together
        for block in final_content:
            if isinstance(block, dict) and "text" in block:
                response_text += block["text"]
            else:
                response_text += str(block)
    else:
        # It's just a simple string
        response_text = str(final_content)
    # ----------------------------------------

    # Display and Save
    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})