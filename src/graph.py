import os
from dotenv import load_dotenv
from .tools import python_analyst, predict_spending_trend

# Import BOTH providers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

load_dotenv()

def get_llm():
    """Selects the LLM based on environment variable."""
    provider = os.getenv("MODEL_PROVIDER", "google").lower()
    
    if provider == "groq":
        print("Using Model: Llama-3.3 (via Groq)")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_retries=2,
            api_key=os.environ["GROQ_API_KEY"]
        )
    else:
        print("Using Model: Gemini-1.5-Flash (via Google)")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_retries=2,
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )

# 1. Load the selected model
llm = get_llm()

# 2. Define the tools
tools = [python_analyst, predict_spending_trend]

# 3. Create the Agent
app = create_react_agent(llm, tools)

def process_query(query: str):
    """
    Main entry point.
    """
    system_instruction = SystemMessage(
        content="""You are a WealthWise financial advisor.
        
        CRITICAL DATA INSTRUCTION:
        You have a pandas DataFrame named `df` ALREADY LOADED into your 'python_analyst' tool.
        - The dataframe contains columns: 'Date', 'Category', 'Amount', 'Description'.
        - You DO NOT need to ask the user for a file. It is already in memory.
        - To answer questions like "How much did I spend on Rent?", execute Python code using `df`.
        
        TOOLS:
        1. 'python_analyst': Use this for math, data aggregation, and specific questions about past transactions.
           Example: `df[df['Category'] == 'Rent']['Amount'].sum()`
        2. 'predict_spending_trend': Use this ONLY for future forecasting.
        
        If a tool returns an error, show the exact error message.
        """
    )
    
    inputs = {"messages": [system_instruction, ("user", query)]}
    result = app.invoke(inputs)
    return result