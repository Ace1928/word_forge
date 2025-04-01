"""
Parser configuration for lexical data parsing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Optional

from word_forge.configs.config_essentials import (  # Type variables
    DATA_ROOT,
    ConfigError,
)
from word_forge.configs.config_types import EnvMapping


@dataclass
class ParserConfig:
    """
    Configuration for lexical data parser.

    Controls data sources, model settings, and resource paths for
    the system's parsing components.

    Attributes:
        data_dir: Base directory for lexical resources
        enable_model: Whether to use language models for examples
        model_name: Custom language model name
        resource_paths: Paths to various lexical resources, relative to data_dir
        ENV_VARS: Mapping of environment variables to config attributes
    """

    # Base directory for lexical resources
    data_dir: str = str(DATA_ROOT)

    # Control language model usage for examples
    enable_model: bool = True

    # Custom language model name (None = use vectorizer's model)
    model_name: Optional[str] = None

    # Resource paths relative to data_dir
    resource_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "openthesaurus": "openthesaurus.jsonl",
            "odict": "odict.json",
            "dbnary": "dbnary.ttl",
            "opendict": "opendict.json",
            "thesaurus": "thesaurus.jsonl",
        }
    )

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_DATA_DIR": ("data_dir", str),
        "WORD_FORGE_ENABLE_MODEL": ("enable_model", bool),
    }

    def get_full_resource_path(self, resource_name: str) -> Path:
        """
        Get absolute path for a resource.

        Args:
            resource_name: The name of the resource as defined in resource_paths

        Returns:
            Absolute path to the resource

        Raises:
            ConfigError: If the resource name is unknown
        """
        if resource_name not in self.resource_paths:
            raise ConfigError(f"Unknown resource: {resource_name}")
        return Path(self.data_dir) / self.resource_paths[resource_name]
