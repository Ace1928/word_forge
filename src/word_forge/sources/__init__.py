"""Authoritative lexical-source metadata and distribution policy."""

from word_forge.sources.registry import (
    CATALOG_NOTICE,
    SOURCE_CATALOG_VERSION,
    BootstrapTier,
    CommercialUse,
    IntegrationStatus,
    LexicalSource,
    LicenseClass,
    SourceNotFoundError,
    get_source,
    iter_sources,
    source_catalog_report,
)

__all__ = [
    "CATALOG_NOTICE",
    "SOURCE_CATALOG_VERSION",
    "BootstrapTier",
    "CommercialUse",
    "IntegrationStatus",
    "LexicalSource",
    "LicenseClass",
    "SourceNotFoundError",
    "get_source",
    "iter_sources",
    "source_catalog_report",
]
