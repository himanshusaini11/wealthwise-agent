import pandas as pd
import joblib
import os
import re
from typing import Union  # <--- NEW IMPORT
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import boto3
from dotenv import load_dotenv

load_dotenv()

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "transactions.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "spending_model.pkl")

# --- 1. INTELLIGENT SCHEMA (The Fix) ---
class ForecastInput(BaseModel):
    """Inputs for the spending prediction tool."""
    
    # CHANGE: Allow int OR str so Groq doesn't block "30"
    days: Union[int, str] = Field(
        description="The number of days to forecast. Examples: '7', '14', '30', 'fortnight'."
    )

    # --- THE MAGIC LAYER ---
    @field_validator('days', mode='before')
    @classmethod
    def parse_natural_language(cls, v):
        """
        Converts the input to an integer.
        """
        # If it's already a number, just return it
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
            
        # If it's text, perform the mapping here
        text = str(v).lower().strip()
        
        # 1. Simple Mapping
        mapping = {
            'fortnight': 14, 'biweek': 14, 'week': 7, 
            'month': 30, 'quarter': 90, 'year': 365
        }
        for key, val in mapping.items():
            if key in text:
                return val
        
        # 2. Regex for things like "2 weeks"
        digits = re.findall(r'\d+', text)
        if digits:
            num = int(digits[0])
            if 'week' in text: return num * 7
            if 'month' in text: return num * 30
            return num
            
        # 3. Last Resort Fallback
        return 7

# --- 2. THE TOOL ---
@tool("predict_spending_trend", args_schema=ForecastInput)
def predict_spending_trend(days: int) -> str:
    """
    Predicts future spending. 
    """
    # Note: Even though input allowed str, the validator guarantees 'days' is int here.
    
    # 1. Ensure Model Exists
    if not os.path.exists(MODEL_PATH):
        try:
            print("Downloading model from S3...")
            s3 = boto3.client('s3', region_name=os.getenv("AWS_DEFAULT_REGION"))
            s3.download_file(os.getenv("S3_BUCKET_NAME"), "spending_model.pkl", MODEL_PATH)
        except Exception as e:
            return f"Error: Model not found. {str(e)}"

    # 2. Load & Predict
    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]
        last_day = artifact["last_day_index"]
        
        future_day_index = last_day + days
        input_df = pd.DataFrame({'Days_Since_Start': [future_day_index]})
        
        predicted_daily_spend = model.predict(input_df)[0]
        total_projected = predicted_daily_spend * days
        
        return (f"PREDICTION COMPLETE: Based on your trend (Day 0 to {last_day}), "
                f"your projected spending for the next {days} days is approx ${total_projected:.2f}.")
    except Exception as e:
        return f"Error running prediction: {str(e)}"

# Initialize Python Analyst
try:
    df = pd.read_csv(DATA_PATH)
    python_analyst = PythonAstREPLTool(locals={"df": df})
    python_analyst.name = "python_analyst"
    python_analyst.description = "A Python shell for analyzing past data. DataFrame 'df' is loaded."
except Exception as e:
    python_analyst = None