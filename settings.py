"""Application settings loaded from environment variables or a .env file.

Copy .env.example to .env and edit the values for local development.
In CI and Docker, set the variables directly in the environment instead.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment: str = "pitch_type_lgbm"
    registered_model_name: str = "PitchTypeClassifier"
    # Base URL of the FastAPI server; used by the Streamlit frontend.
    api_url: str = "http://localhost:8001"


settings = Settings()
