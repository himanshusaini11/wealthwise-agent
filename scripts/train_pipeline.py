# FILE: scripts/train_pipeline.py
import os

import boto3
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    # Exclude fixed monthly bills — they are not trend-driven discretionary spending
    FIXED_CATEGORIES = {'Rent', 'Subscriptions'}
    expenses = df[(df['Amount'] < 0) & (~df['Category'].isin(FIXED_CATEGORIES))].copy()
    expenses['Amount'] = expenses['Amount'].abs()

    daily = expenses.groupby('Date')['Amount'].sum().reset_index()

    start_date = daily['Date'].min()
    daily['Days_Since_Start'] = (daily['Date'] - start_date).dt.days
    daily['day_of_week'] = pd.to_datetime(daily['Date']).dt.dayofweek
    daily['month'] = pd.to_datetime(daily['Date']).dt.month
    daily['is_weekend'] = (daily['day_of_week'] >= 5).astype(int)

    X = daily[['Days_Since_Start', 'day_of_week', 'month', 'is_weekend']]
    y = daily['Amount']

    # 2. Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train Model
    print("Training model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)

    # 4. Validation metrics
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 5. Quality gate
    # Default -1.0 allows synthetic data during development.
    # Set R2_THRESHOLD=0.3 in production for a meaningful quality gate.
    R2_THRESHOLD = float(os.getenv("R2_THRESHOLD", "-1.0"))
    if r2 < R2_THRESHOLD:
        raise ValueError(
            f"Model quality gate failed: R²={r2:.4f} is below "
            f"threshold {R2_THRESHOLD}. Check your training data."
        )

    # 6. Save Artifacts
    artifact = {
        "model": pipeline,
        "start_date": str(daily['Date'].min()),
        "last_day_index": int(daily['Days_Since_Start'].max()),
        "feature_names": ['Days_Since_Start', 'day_of_week', 'month', 'is_weekend'],
    }

    print(f"Saving artifact to {MODEL_PATH}...")
    joblib.dump(artifact, MODEL_PATH)

    # 7. MLflow Logging
    db_path = os.path.join(ROOT_DIR, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("WealthWise_Forecast")

    with mlflow.start_run():
        mlflow.log_param("rows", len(df))
        mlflow.log_metric("r2_score", round(r2, 4))
        mlflow.log_metric("mae", round(mae, 4))
        mlflow.log_metric("rmse", round(rmse, 4))
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.sklearn.log_model(pipeline, "model")
        print("Logged to MLflow.")

        print("\n=== Training Summary ===")
        print(f"Train samples : {len(X_train)}")
        print(f"Test samples  : {len(X_test)}")
        print(f"R² score      : {r2:.4f}")
        print(f"MAE           : ${mae:.2f}/day")
        print(f"RMSE          : ${rmse:.2f}/day")
        print(f"Quality gate  : {'PASSED' if r2 >= R2_THRESHOLD else 'FAILED'}")
        print(f"MLflow run    : {mlflow.active_run().info.run_id}")
        print("========================\n")

    # 8. S3 Upload
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
