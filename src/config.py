from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Literal
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_provider: Literal["claude-haiku", "claude-sonnet", "gemini", "groq"] = "claude-haiku"

    # API keys — required only if that provider is selected
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    google_api_key: str = ""

    # AWS — optional, only needed for S3 model fallback
    aws_default_region: str = "us-east-1"
    s3_bucket_name: str = ""

    @field_validator("anthropic_api_key")
    @classmethod
    def validate_anthropic_key(cls, v, info):
        if info.data.get("model_provider") in ("claude-haiku", "claude-sonnet") and not v:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when MODEL_PROVIDER is claude-haiku or claude-sonnet"
            )
        return v

    @field_validator("groq_api_key")
    @classmethod
    def validate_groq_key(cls, v, info):
        if info.data.get("model_provider") == "groq" and not v:
            raise ValueError(
                "GROQ_API_KEY is required when MODEL_PROVIDER=groq"
            )
        return v

    @field_validator("google_api_key")
    @classmethod
    def validate_google_key(cls, v, info):
        if info.data.get("model_provider") == "gemini" and not v:
            raise ValueError(
                "GOOGLE_API_KEY is required when MODEL_PROVIDER=gemini"
            )
        return v


def get_settings() -> Settings:
    settings = Settings()
    logger.info(
        "Configuration loaded",
        extra={"model_provider": settings.model_provider},
    )
    return settings