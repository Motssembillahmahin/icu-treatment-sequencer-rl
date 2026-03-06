"""Application settings via Pydantic v2 BaseSettings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: str = Field("development", description="development | production | test")
    log_level: str = Field("INFO", description="Logging level")

    # API
    api_host: str = Field("0.0.0.0", description="Server bind host")
    api_port: int = Field(8000, description="Server port")

    # Paths
    models_dir: Path = Field(Path("models"), description="Directory for saved model artifacts")
    data_dir: Path = Field(Path("data"), description="Directory for data/SQLite DBs")
    runs_dir: Path = Field(Path("runs"), description="TensorBoard log directory")

    # Training defaults
    default_agent: str = Field("ppo", description="Default RL algorithm")
    default_config: Path = Field(
        Path("configs/hyperparams/ppo_default.yaml"),
        description="Default hyperparameter YAML",
    )

    @property
    def episodes_db(self) -> Path:
        return self.data_dir / "episodes" / "episodes.db"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_db.parent.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
