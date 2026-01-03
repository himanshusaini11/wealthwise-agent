import pandas as pd
import joblib
import os
from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import boto3
from dotenv import load_dotenv

load_dotenv()

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "transactions.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "spending_model.pkl")
# ------------------

@tool
def predict_spending_trend(days: int) -> str:
    """
    Predicts future spending.
    Args:
        days: The number of days into the future to predict (e.g., 7 for next week).
    """
    # 1. Ensure Model Exists
    if not os.path.exists(MODEL_PATH):
        try:
            print("Downloading model from S3...")
            s3 = boto3.client('s3', region_name=os.getenv("AWS_DEFAULT_REGION"))
            s3.download_file(os.getenv("S3_BUCKET_NAME"), "spending_model.pkl", MODEL_PATH)
        except Exception as e:
            return f"Error: Model not found and S3 download failed. {str(e)}"

    # 2. Load Artifact & Predict
    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]
        last_day = artifact["last_day_index"]
        
        future_day_index = last_day + days
        input_df = pd.DataFrame({'Days_Since_Start': [future_day_index]})
        
        predicted_daily_spend = model.predict(input_df)[0]
        total_projected = predicted_daily_spend * days
        
        return (f"Based on your trend (Day 0 to {last_day}), "
                f"your projected spending for the next {days} days is approx ${total_projected:.2f}.")
    except Exception as e:
        return f"Error running prediction: {str(e)}"

# Initialize Python Analyst with EXPLICIT Name
try:
    df = pd.read_csv(DATA_PATH)
    # We rename the tool so the Agent knows exactly what to call
    python_analyst = PythonAstREPLTool(locals={"df": df})
    python_analyst.name = "python_analyst"
    python_analyst.description = (
        "A Python shell for analyzing data. "
        "The dataframe 'df' is ALREADY LOADED in memory. "
        "Use this tool to run pandas code on 'df' to answer questions."
    )
except Exception as e:
    print(f"Warning: Could not load data for python_analyst. {e}")
    python_analyst = None