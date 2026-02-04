"""Configuration management for PDF to PPT converter."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM service configuration."""

    base_url: str = Field(default="", description="Base URL of the LLM API service")
    api_key: str = Field(default="", description="API key for authentication")
    model_name: str = Field(default="", description="Model name to use")
    enabled: bool = Field(default=False, description="Enable LLM enhancement")
    batch_size: int = Field(default=5, description="Pages per batch")
    timeout: int = Field(default=60, description="API timeout in seconds")

    def is_configured(self) -> bool:
        """Check if LLM is properly configured."""
        return bool(self.base_url and self.model_name)

    def get_model_info(self) -> str:
        """Get model info for display."""
        if not self.is_configured():
            return "LLM: Not configured"
        return f"LLM: {self.model_name} @ {self.base_url}"


class Config(BaseModel):
    """Main configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load configuration from file.

        Args:
            config_path: Path to config file. If None, looks for config.yaml in project root.

        Returns:
            Config instance
        """
        if config_path is None:
            # Find project root (look for config.yaml)
            current_dir = Path(__file__).parent
            while current_dir.parent != current_dir:
                config_path = current_dir / "config.yaml"
                if config_path.exists():
                    break
                current_dir = current_dir.parent
            else:
                # No config file found, return defaults
                return cls()

        if not config_path.exists():
            return cls()

        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Parse LLM config
            llm_data = data.get("llm", {})
            llm_config = LLMConfig(**llm_data)

            return cls(llm=llm_config)
        except ImportError:
            # yaml not available, return defaults
            return cls()
        except Exception:
            # Error parsing config, return defaults
            return cls()

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.llm


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """Get or load configuration."""
    global _config
    if _config is None:
        _config = Config.load(config_path)
    return _config


def reload_config(config_path: Optional[Path] = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = Config.load(config_path)
    return _config
