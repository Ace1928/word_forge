"""Tests for lexical source governance and machine-readable discovery."""

from __future__ import annotations

import json

import pytest

from word_forge.sources import (
    BootstrapTier,
    CommercialUse,
    LicenseClass,
    SourceNotFoundError,
    get_source,
    iter_sources,
    source_catalog_report,
)


def test_catalog_is_stable_unique_and_serializable() -> None:
    sources = list(iter_sources())
    identifiers = [source.source_id for source in sources]

    assert identifiers == sorted(identifiers)
    assert len(identifiers) == len(set(identifiers))
    assert len(sources) >= 10
    assert json.loads(json.dumps(source_catalog_report()))["count"] == len(sources)


def test_unattended_catalog_excludes_restricted_policy_tiers() -> None:
    sources = list(iter_sources(unattended_only=True))

    assert sources
    assert all(source.unattended_eligible for source in sources)
    assert all(
        source.bootstrap_tier in {BootstrapTier.CORE, BootstrapTier.PERMISSIVE}
        for source in sources
    )
    assert "dbnary" not in {source.source_id for source in sources}
    assert "unimorph" not in {source.source_id for source in sources}


def test_catalog_preserves_non_uniform_license_boundaries() -> None:
    dbnary = get_source("DBNARY")
    unimorph = get_source("unimorph")
    wikidata = get_source("wikidata-lexemes")

    assert dbnary.license_class is LicenseClass.ATTRIBUTION_SHARE_ALIKE
    assert not dbnary.unattended_eligible
    assert unimorph.license_class is LicenseClass.PER_DATASET
    assert unimorph.commercial_use is CommercialUse.PER_DATASET_REVIEW
    assert wikidata.license_class is LicenseClass.PUBLIC_DOMAIN
    assert wikidata.unattended_eligible


def test_unknown_source_error_lists_available_identifiers() -> None:
    with pytest.raises(SourceNotFoundError, match="princeton-wordnet"):
        get_source("not-a-real-source")
