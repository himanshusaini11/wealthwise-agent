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
        
        CRITICAL INSTRUCTIONS:
        1. DATA: You have a dataframe `df` loaded in 'python_analyst'. Use it for past data questions.
        2. FORECASTING: Use 'predict_spending_trend' for future questions.
           - ARGUMENT FORMAT: You MUST convert written numbers to DIGITS.
             (e.g., if user says "thirty days", input "30". If "two weeks", input "14").
        3. STOPPING: Once you receive a "PREDICTION COMPLETE" message from a tool, STOP immediately and report the result to the user.
        """
    )
    
    inputs = {"messages": [system_instruction, ("user", query)]}
    
    # Keep recursion limit at 50
    result = app.invoke(inputs, config={"recursion_limit": 50})
    return result