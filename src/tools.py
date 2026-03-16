import logging
import os
import re
import time
from datetime import timedelta
from typing import Union

import boto3
import joblib
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from pydantic import BaseModel, Field, field_validator

load_dotenv()
logger = logging.getLogger(__name__)

# --- Paths ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, "data", "transactions.csv")
MODEL_PATH = os.path.join(_ROOT, "models", "spending_model.pkl")


# --- Input schema ---
class ForecastInput(BaseModel):
    """Input schema for predict_spending_trend."""

    days: Union[int, str] = Field(
        description=(
            "Number of days to forecast. "
            "Accepts integers or natural language: '14', 'fortnight', '2 weeks'."
        )
    )

    @field_validator("days", mode="before")
    @classmethod
    def parse_natural_language(cls, v) -> int:
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())

        text = str(v).lower().strip()

        # STEP 1: Regex first — handles "2 weeks", "3 months", "10 days"
        digits = re.findall(r"\d+", text)
        if digits:
            num = int(digits[0])
            if "week" in text:
                return num * 7
            if "month" in text:
                return num * 30
            return num

        # STEP 2: Exact word mapping — handles "week", "fortnight", "month" alone
        exact = {
            "fortnight": 14,
            "biweek": 14,
            "week": 7,
            "month": 30,
            "quarter": 90,
            "year": 365,
        }
        for key, val in exact.items():
            if key in text:
                logger.debug("NL parse matched exact key", extra={"key": key, "value": val})
                return val

        # STEP 3: Fallback
        logger.warning(
            "parse_natural_language: could not parse input, defaulting to 7",
            extra={"raw_input": v},
        )
        return 7


def _load_model():
    """Loads model from disk, falls back to S3 download if missing."""
    if not os.path.exists(MODEL_PATH):
        logger.warning("Model not found locally, attempting S3 download")
        try:
            s3 = boto3.client(
                "s3",
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
            bucket = os.getenv("S3_BUCKET_NAME", "")
            if not bucket:
                raise ValueError("S3_BUCKET_NAME env var is not set")
            s3.download_file(bucket, "spending_model.pkl", MODEL_PATH)
            logger.info("Model downloaded from S3 successfully")
        except Exception as e:
            logger.exception("S3 model download failed")
            raise RuntimeError(f"Model unavailable: {e}") from e

    return joblib.load(MODEL_PATH)


@tool("predict_spending_trend", args_schema=ForecastInput)
def predict_spending_trend(days: int) -> str:
    """
    Predicts total spending over the next N days using a trained ML model.
    Returns a plain-English summary prefixed with PREDICTION COMPLETE.
    """
    logger.info("predict_spending_trend called", extra={"days": days})
    start = time.perf_counter()

    try:
        artifact = _load_model()
        model = artifact["model"]
        last_day = artifact["last_day_index"]

        future_index = last_day + days
        start_date = pd.to_datetime(artifact["start_date"])
        future_date = start_date + pd.DateOffset(days=future_index)

        input_df = pd.DataFrame({
            'Days_Since_Start': [future_index],
            'day_of_week': [future_date.dayofweek],
            'month': [future_date.month],
            'is_weekend': [int(future_date.dayofweek >= 5)],
        })
        daily = model.predict(input_df)[0]
        total = daily * days

        elapsed = time.perf_counter() - start
        logger.info(
            "predict_spending_trend complete",
            extra={"days": days, "total": round(total, 2), "elapsed_s": round(elapsed, 3)},
        )
        total_str = f"{total:.2f}"
        daily_str = f"{daily:.2f}"
        return (
            f"PREDICTION COMPLETE: Projected spending over the next {days} days "
            f"is approximately ${total_str} (avg ${daily_str}/day)."
        )
    except RuntimeError as e:
        logger.error("predict_spending_trend failed", extra={"error": str(e)})
        return f"Prediction unavailable: {str(e)}"
    except Exception as e:
        logger.exception("Unexpected error in predict_spending_trend")
        return f"Prediction failed due to an unexpected error: {str(e)}"


# --- Python analyst tool ---
def _build_python_analyst() -> PythonAstREPLTool | None:
    try:
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Category'] = df['Category'].str.strip().str.title()
        tool_instance = PythonAstREPLTool(locals={"df": df})
        tool_instance.name = "python_analyst"
        tool_instance.description = (
            "A Python shell for analysing past transaction data. "
            "DataFrame 'df' is pre-loaded with columns: Date, Category, Amount, Description."
        )
        logger.info(
            "python_analyst initialised",
            extra={"rows": len(df), "columns": list(df.columns)},
        )
        return tool_instance
    except Exception as e:
        logger.exception("Failed to initialise python_analyst", extra={"error": str(e)})
        return None