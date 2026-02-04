"""Configuration management for PDF to PPT converter."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# LLM Presets configuration
LLM_PRESETS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "",
        "model_name": "qwen3-vl:4b",
        "no_proxy": "localhost",
        "use_enhancement": False,  # CPU inference is too slow
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model_name": "gpt-4o",
        "no_proxy": "",
        "use_enhancement": True,  # Cloud API is fast enough
    },
}


class LLMConfig(BaseModel):
    """LLM service configuration."""

    preset: str = Field(default="custom", description="Preset name (ollama, openai, custom)")
    base_url: str = Field(default="", description="Base URL of the LLM API service")
    api_key: str = Field(default="", description="API key for authentication")
    model_name: str = Field(default="", description="Model name to use")
    enabled: bool = Field(default=False, description="Enable LLM for OCR")
    use_enhancement: bool = Field(default=False, description="Enable LLM layout enhancement (slow on CPU)")
    no_proxy: str = Field(default="", description="No proxy settings for internal services")
    batch_size: int = Field(default=5, description="Pages per batch")
    timeout: int = Field(default=60, description="API timeout in seconds")

    def is_configured(self) -> bool:
        """Check if LLM is properly configured."""
        return bool(self.base_url and self.model_name)

    def get_model_info(self) -> str:
        """Get model info for display."""
        if not self.is_configured():
            return "LLM: Not configured"
        preset_info = f" [{self.preset}]" if self.preset != "custom" else ""
        return f"LLM: {self.model_name}{preset_info} @ {self.base_url}"

    def apply_preset(self) -> "LLMConfig":
        """
        Apply preset configuration if set.

        Returns:
            Self with preset applied
        """
        if self.preset in LLM_PRESETS:
            preset_config = LLM_PRESETS[self.preset]
            # Only use preset values if custom values are empty
            if not self.base_url:
                self.base_url = preset_config["base_url"]
            if not self.model_name and preset_config["model_name"]:
                self.model_name = preset_config["model_name"]
            if not self.no_proxy and preset_config["no_proxy"]:
                self.no_proxy = preset_config["no_proxy"]
            # api_key from preset only if it's not empty (for security)
            if preset_config["api_key"] and not self.api_key:
                self.api_key = preset_config["api_key"]
            # use_enhancement from preset
            self.use_enhancement = preset_config["use_enhancement"]
        return self


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

            # Apply preset if set
            llm_config = llm_config.apply_preset()

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
