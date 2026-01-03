# FILE: scripts/train_pipeline.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
import boto3

load_dotenv()

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "transactions.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "spending_model.pkl")
# ------------------

def run_training():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing data at {DATA_PATH}. Run generate_data.py first.")
        
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])

    # 1. Feature Engineering
    # Filter for expenses only (negative values)
    expenses = df[df['Amount'] < 0].copy()
    expenses['Amount'] = expenses['Amount'].abs()
    
    # Aggregate by day
    daily_spend = expenses.groupby('Date')['Amount'].sum().reset_index()
    
    # Create the 'X' feature: Days since the first transaction
    start_date = daily_spend['Date'].min()
    daily_spend['Days_Since_Start'] = (daily_spend['Date'] - start_date).dt.days
    
    X = daily_spend[['Days_Since_Start']]  # Double brackets to keep it a DataFrame
    y = daily_spend['Amount']

    # 2. Train Model
    print("Training model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    pipeline.fit(X, y)

    # 3. Save Artifacts (Model + Metadata)
    # We create a dictionary to save both the model AND the start_date reference
    artifact = {
        "model": pipeline,
        "start_date": start_date,
        "last_day_index": daily_spend['Days_Since_Start'].max()
    }
    
    print(f"Saving artifact to {MODEL_PATH}...")
    joblib.dump(artifact, MODEL_PATH)

    # 4. MLflow Logging (SQLite)
    db_path = os.path.join(ROOT_DIR, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("WealthWise_Forecast")
    
    with mlflow.start_run():
        mlflow.log_param("rows", len(df))
        mlflow.sklearn.log_model(pipeline, "model")
        print("Logged to MLflow.")

    # 5. S3 Upload
    print("Uploading to S3...")
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION")
        )
        bucket = os.getenv("S3_BUCKET_NAME")
        if bucket:
            s3.upload_file(MODEL_PATH, bucket, "spending_model.pkl")
            print(f"Uploaded to S3: {bucket}")
    except Exception as e:
        print(f"S3 Upload Warning: {e}")

if __name__ == "__main__":
    run_training()