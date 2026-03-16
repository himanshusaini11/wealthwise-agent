"""Tests for src/tools.py"""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.tools import ForecastInput, _build_python_analyst, predict_spending_trend


# ---------------------------------------------------------------------------
# 1. parse_natural_language validator
# ---------------------------------------------------------------------------

class TestParseNaturalLanguage:
    def test_integer_input(self):
        assert ForecastInput(days=7).days == 7

    def test_digit_string(self):
        assert ForecastInput(days="14").days == 14

    def test_fortnight(self):
        assert ForecastInput(days="fortnight").days == 14

    def test_week(self):
        assert ForecastInput(days="week").days == 7

    def test_month(self):
        assert ForecastInput(days="month").days == 30

    def test_quarter(self):
        assert ForecastInput(days="quarter").days == 90

    def test_year(self):
        assert ForecastInput(days="year").days == 365

    def test_two_weeks(self):
        assert ForecastInput(days="2 weeks").days == 14

    def test_three_months(self):
        assert ForecastInput(days="3 months").days == 90

    def test_ten_days(self):
        assert ForecastInput(days="10 days").days == 10

    def test_biweek(self):
        assert ForecastInput(days="biweek").days == 14

    def test_unparseable_fallback(self):
        assert ForecastInput(days="banana").days == 7

    def test_uppercase(self):
        # "TWO WEEKS" lowercases to "two weeks"; no digits, so regex step is
        # skipped; "week" hits the exact-match returning 7. Written numbers
        # ("two") are not parsed — this is a known limitation.
        assert ForecastInput(days="TWO WEEKS").days == 7


# ---------------------------------------------------------------------------
# 2. predict_spending_trend with mocked model
# ---------------------------------------------------------------------------

class TestPredictSpendingTrend:
    def test_prediction_output(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [50.0]
        fake_artifact = {"model": mock_model, "last_day_index": 90, "start_date": "2025-12-16"}

        with patch("src.tools.os.path.exists", return_value=True), \
             patch("src.tools.joblib.load", return_value=fake_artifact):
            response = predict_spending_trend.invoke({"days": 14})

        assert response.startswith("PREDICTION COMPLETE")
        assert "14 days" in response
        assert "$700.00" in response  # 50.0 * 14 = 700.00


# ---------------------------------------------------------------------------
# 3. _build_python_analyst
# ---------------------------------------------------------------------------

class TestBuildPythonAnalyst:
    def test_returns_tool(self):
        fake_df = pd.DataFrame({
            "Date": ["2026-01-01", "2026-01-02"],
            "Category": ["food", "transport"],
            "Amount": [-10.0, -5.0],
            "Description": ["lunch", "bus"],
        })
        with patch("src.tools.pd.read_csv", return_value=fake_df):
            tool_instance = _build_python_analyst()

        assert tool_instance is not None
        assert tool_instance.name == "python_analyst"
        assert "df" in tool_instance.locals

    def test_returns_none_on_failure(self):
        with patch("src.tools.pd.read_csv", side_effect=FileNotFoundError):
            tool_instance = _build_python_analyst()
        assert tool_instance is None


# ---------------------------------------------------------------------------
# 4. S3 fallback path in predict_spending_trend
# ---------------------------------------------------------------------------

class TestPredictSpendingTrendS3Fallback:
    def test_s3_unavailable_returns_error_string(self):
        with patch("src.tools.os.path.exists", return_value=False), \
             patch("src.tools.boto3.client", side_effect=Exception("S3 unavailable")):
            response = predict_spending_trend.invoke({"days": 7})
        assert "Prediction unavailable" in response


# ---------------------------------------------------------------------------
# 5. _load_model failure path
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_raises_runtime_error_when_model_missing(self):
        from src.tools import _load_model
        with patch("src.tools.os.path.exists", return_value=False), \
             patch("src.tools.boto3.client", side_effect=Exception("Network error")):
            with pytest.raises(RuntimeError, match="Model unavailable"):
                _load_model()
