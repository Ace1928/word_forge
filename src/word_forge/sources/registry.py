"""Machine-readable governance for external lexical data sources.

Word Forge is MIT licensed, but lexical datasets keep their own licenses.  This
module deliberately separates source metadata from importer implementation so
that every download path can enforce the same commercial-use, attribution, and
redistribution boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, Mapping, Tuple

SOURCE_CATALOG_VERSION = 1
CATALOG_NOTICE = (
    "Source metadata is operational guidance, not legal advice. The license "
    "and notices shipped with the exact downloaded snapshot are authoritative."
)


class LicenseClass(str, Enum):
    """Broad obligations used to isolate downloaded data correctly."""

    PUBLIC_DOMAIN = "public-domain"
    PERMISSIVE = "permissive"
    ATTRIBUTION_SHARE_ALIKE = "attribution-share-alike"
    COPYLEFT = "copyleft"
    PER_DATASET = "per-dataset"


class CommercialUse(str, Enum):
    """Whether the catalog can establish commercial-use eligibility."""

    ALLOWED = "allowed"
    ALLOWED_WITH_TERMS = "allowed-with-terms"
    PER_DATASET_REVIEW = "per-dataset-review"


class IntegrationStatus(str, Enum):
    """Current relationship between a source and Word Forge."""

    BUILT_IN = "built-in"
    OPTIONAL = "optional"
    PLANNED = "planned"
    EXTERNAL_RUNTIME = "external-runtime"


class BootstrapTier(str, Enum):
    """Policy tier controlling future automated source installation."""

    CORE = "core"
    PERMISSIVE = "permissive"
    SHARE_ALIKE_OPT_IN = "share-alike-opt-in"
    PER_DATASET_REVIEW = "per-dataset-review"
    EXTERNAL_RUNTIME = "external-runtime"

    @property
    def unattended_eligible(self) -> bool:
        """Return whether this tier may be enabled without license prompts."""

        return self in {BootstrapTier.CORE, BootstrapTier.PERMISSIVE}


@dataclass(frozen=True, slots=True)
class LexicalSource:
    """Stable metadata and policy for one lexical source family."""

    source_id: str
    name: str
    homepage: str
    license_name: str
    license_url: str
    license_class: LicenseClass
    commercial_use: CommercialUse
    integration_status: IntegrationStatus
    bootstrap_tier: BootstrapTier
    language_scope: str
    data_types: Tuple[str, ...]
    attribution: str
    redistribution: str
    notes: str = ""

    def __post_init__(self) -> None:
        """Reject incomplete metadata before it reaches an importer."""

        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.source_id) is None:
            raise ValueError(f"Invalid lexical source id: {self.source_id!r}")
        for field_name, value in (
            ("name", self.name),
            ("license_name", self.license_name),
            ("language_scope", self.language_scope),
            ("attribution", self.attribution),
            ("redistribution", self.redistribution),
        ):
            if not value.strip():
                raise ValueError(
                    f"Lexical source {self.source_id!r} has no {field_name}"
                )
        for field_name, value in (
            ("homepage", self.homepage),
            ("license_url", self.license_url),
        ):
            if not value.startswith("https://"):
                raise ValueError(
                    f"Lexical source {self.source_id!r} {field_name} must use HTTPS"
                )
        if not self.data_types or any(not item.strip() for item in self.data_types):
            raise ValueError(
                f"Lexical source {self.source_id!r} must declare data types"
            )

    @property
    def unattended_eligible(self) -> bool:
        """Return whether unattended bootstrap may consider this source."""

        return self.bootstrap_tier.unattended_eligible

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "id": self.source_id,
            "name": self.name,
            "homepage": self.homepage,
            "license": {
                "name": self.license_name,
                "url": self.license_url,
                "class": self.license_class.value,
                "commercial_use": self.commercial_use.value,
                "attribution": self.attribution,
                "redistribution": self.redistribution,
            },
            "integration_status": self.integration_status.value,
            "bootstrap_tier": self.bootstrap_tier.value,
            "unattended_eligible": self.unattended_eligible,
            "language_scope": self.language_scope,
            "data_types": list(self.data_types),
            "notes": self.notes,
        }


_SOURCES: Tuple[LexicalSource, ...] = (
    LexicalSource(
        source_id="cmudict",
        name="CMU Pronouncing Dictionary",
        homepage="https://github.com/cmusphinx/cmudict",
        license_name="CMUdict unrestricted-use notice",
        license_url="https://github.com/cmusphinx/cmudict/blob/master/LICENSE",
        license_class=LicenseClass.PERMISSIVE,
        commercial_use=CommercialUse.ALLOWED,
        integration_status=IntegrationStatus.BUILT_IN,
        bootstrap_tier=BootstrapTier.CORE,
        language_scope="English (United States)",
        data_types=("pronunciations", "phonemes", "stress"),
        attribution="Acknowledge Carnegie Mellon University as requested.",
        redistribution="Retain the upstream notice and origin acknowledgement.",
        notes="Word Forge reads CMUdict through NLTK and marks derived IPA records.",
    ),
    LexicalSource(
        source_id="dbnary",
        name="DBnary",
        homepage="https://kaiko.getalp.org/about-dbnary/",
        license_name="Creative Commons Attribution-ShareAlike 3.0",
        license_url="https://creativecommons.org/licenses/by-sa/3.0/",
        license_class=LicenseClass.ATTRIBUTION_SHARE_ALIKE,
        commercial_use=CommercialUse.ALLOWED_WITH_TERMS,
        integration_status=IntegrationStatus.OPTIONAL,
        bootstrap_tier=BootstrapTier.SHARE_ALIKE_OPT_IN,
        language_scope="Multiple Wiktionary language editions",
        data_types=(
            "lexemes",
            "senses",
            "translations",
            "relations",
            "morphology",
            "etymology",
        ),
        attribution="Attribute DBnary and the originating Wiktionary content.",
        redistribution="Keep imported data and adaptations under applicable BY-SA terms.",
        notes="Download metadata may specify additional or newer terms and controls.",
    ),
    LexicalSource(
        source_id="epitran",
        name="Epitran",
        homepage="https://github.com/dmort27/epitran",
        license_name="MIT License",
        license_url="https://github.com/dmort27/epitran/blob/master/LICENSE.txt",
        license_class=LicenseClass.PERMISSIVE,
        commercial_use=CommercialUse.ALLOWED_WITH_TERMS,
        integration_status=IntegrationStatus.OPTIONAL,
        bootstrap_tier=BootstrapTier.PERMISSIVE,
        language_scope="Language-and-script-specific mappings",
        data_types=("grapheme-to-phoneme mappings", "IPA transcriptions"),
        attribution="Retain the MIT copyright and permission notice.",
        redistribution="May be redistributed subject to the MIT notice.",
        notes="English transcription additionally requires Flite.",
    ),
    LexicalSource(
        source_id="espeak-ng",
        name="eSpeak NG",
        homepage="https://github.com/espeak-ng/espeak-ng",
        license_name="GNU General Public License v3 or later",
        license_url="https://github.com/espeak-ng/espeak-ng/blob/master/COPYING",
        license_class=LicenseClass.COPYLEFT,
        commercial_use=CommercialUse.ALLOWED_WITH_TERMS,
        integration_status=IntegrationStatus.EXTERNAL_RUNTIME,
        bootstrap_tier=BootstrapTier.EXTERNAL_RUNTIME,
        language_scope="More than one hundred languages and accents",
        data_types=("pronunciation rules", "phoneme inventories", "IPA output"),
        attribution="Preserve upstream copyright and GPL notices.",
        redistribution="Distribute eSpeak NG itself only under its GPL terms.",
        notes="Keep the executable/runtime boundary explicit in MIT distributions.",
    ),
    LexicalSource(
        source_id="kaikki-wiktionary",
        name="Kaikki Wiktionary extracts",
        homepage="https://kaikki.org/dictionary/",
        license_name="Originating Wiktionary content terms",
        license_url="https://dumps.wikimedia.org/legal.html",
        license_class=LicenseClass.ATTRIBUTION_SHARE_ALIKE,
        commercial_use=CommercialUse.ALLOWED_WITH_TERMS,
        integration_status=IntegrationStatus.PLANNED,
        bootstrap_tier=BootstrapTier.SHARE_ALIKE_OPT_IN,
        language_scope="Wiktionary languages represented by each extract",
        data_types=(
            "lexemes",
            "senses",
            "forms",
            "pronunciations",
            "relations",
            "translations",
        ),
        attribution="Preserve Wiktionary attribution and source-page provenance.",
        redistribution="Isolate extracts and adaptations under applicable source terms.",
        notes="Capture the exact extract's notices; the tool license is not the data license.",
    ),
    LexicalSource(
        source_id="open-multilingual-wordnet",
        name="Open Multilingual Wordnet",
        homepage="https://omwn.org/",
        license_name="Component wordnet licenses",
        license_url="https://globalwordnet.github.io/resources/wordnets-in-the-world",
        license_class=LicenseClass.PER_DATASET,
        commercial_use=CommercialUse.PER_DATASET_REVIEW,
        integration_status=IntegrationStatus.OPTIONAL,
        bootstrap_tier=BootstrapTier.PER_DATASET_REVIEW,
        language_scope="Multiple independently licensed wordnets",
        data_types=("synsets", "senses", "semantic relations", "cross-lingual links"),
        attribution="Cite each originating wordnet and the aggregation when applicable.",
        redistribution="Enforce the license declared by every selected component.",
        notes=(
            "NLTK OMW 2.0 is installed only after explicit license acknowledgement; "
            "never infer one blanket license for its component wordnets."
        ),
    ),
    LexicalSource(
        source_id="panlex",
        name="PanLex",
        homepage="https://panlex.org/",
        license_name="CC0 1.0 Universal",
        license_url="https://panlex.org/license",
        license_class=LicenseClass.PUBLIC_DOMAIN,
        commercial_use=CommercialUse.ALLOWED,
        integration_status=IntegrationStatus.PLANNED,
        bootstrap_tier=BootstrapTier.PERMISSIVE,
        language_scope="Panlingual translation expressions",
        data_types=("expressions", "translations", "varieties", "meaning links"),
        attribution="Citation is requested but not required by CC0.",
        redistribution="May be copied, modified, and distributed, including commercially.",
        notes="Prefer versioned snapshots with recorded checksums.",
    ),
    LexicalSource(
        source_id="princeton-wordnet",
        name="Princeton WordNet 3.0",
        homepage="https://wordnet.princeton.edu/",
        license_name="Princeton WordNet 3.0 License",
        license_url="https://wordnet.princeton.edu/license-and-commercial-use",
        license_class=LicenseClass.PERMISSIVE,
        commercial_use=CommercialUse.ALLOWED_WITH_TERMS,
        integration_status=IntegrationStatus.BUILT_IN,
        bootstrap_tier=BootstrapTier.CORE,
        language_scope="English",
        data_types=("synsets", "definitions", "examples", "semantic relations"),
        attribution="Preserve the WordNet copyright, license, and disclaimer.",
        redistribution="Include the required notice on copies and modifications.",
        notes="Word Forge accesses this source through NLTK.",
    ),
    LexicalSource(
        source_id="unicode-cldr",
        name="Unicode CLDR",
        homepage="https://cldr.unicode.org/",
        license_name="Unicode License v3",
        license_url="https://www.unicode.org/license.txt",
        license_class=LicenseClass.PERMISSIVE,
        commercial_use=CommercialUse.ALLOWED_WITH_TERMS,
        integration_status=IntegrationStatus.PLANNED,
        bootstrap_tier=BootstrapTier.PERMISSIVE,
        language_scope="Unicode locales and writing systems",
        data_types=("locale metadata", "script metadata", "transforms", "annotations"),
        attribution="Retain the Unicode copyright and license notice.",
        redistribution="May be redistributed subject to the Unicode License.",
        notes="Pin stable CLDR releases rather than mutable development snapshots.",
    ),
    LexicalSource(
        source_id="unicode-character-database",
        name="Unicode Character Database",
        homepage="https://www.unicode.org/ucd/",
        license_name="Unicode License v3",
        license_url="https://www.unicode.org/license.txt",
        license_class=LicenseClass.PERMISSIVE,
        commercial_use=CommercialUse.ALLOWED_WITH_TERMS,
        integration_status=IntegrationStatus.BUILT_IN,
        bootstrap_tier=BootstrapTier.CORE,
        language_scope="Unicode characters and scripts",
        data_types=("character properties", "normalization", "grapheme properties"),
        attribution="Retain the Unicode copyright and license notice when redistributed.",
        redistribution="May be redistributed subject to the Unicode License.",
        notes="The core currently consumes the Python runtime's Unicode tables.",
    ),
    LexicalSource(
        source_id="unimorph",
        name="UniMorph",
        homepage="https://unimorph.github.io/",
        license_name="Per-language dataset licenses",
        license_url="https://unimorph.github.io/",
        license_class=LicenseClass.PER_DATASET,
        commercial_use=CommercialUse.PER_DATASET_REVIEW,
        integration_status=IntegrationStatus.PLANNED,
        bootstrap_tier=BootstrapTier.PER_DATASET_REVIEW,
        language_scope="Per-language morphology repositories",
        data_types=("lemmas", "inflected forms", "morphological features"),
        attribution="Preserve the source and license declared by each language repository.",
        redistribution="Evaluate and enforce every selected repository's own license.",
        notes="Many repositories use BY-SA, but the catalog must not assume uniform terms.",
    ),
    LexicalSource(
        source_id="wikidata-lexemes",
        name="Wikidata Lexemes",
        homepage="https://www.wikidata.org/wiki/Wikidata:Lexicographical_data",
        license_name="CC0 1.0 Universal",
        license_url="https://dumps.wikimedia.org/legal.html",
        license_class=LicenseClass.PUBLIC_DOMAIN,
        commercial_use=CommercialUse.ALLOWED,
        integration_status=IntegrationStatus.PLANNED,
        bootstrap_tier=BootstrapTier.PERMISSIVE,
        language_scope="Multilingual structured lexicographical data",
        data_types=("lexemes", "lemmas", "forms", "senses", "statements"),
        attribution="Attribution is appreciated but not required by CC0.",
        redistribution="Structured Lexeme-namespace data may be reused under CC0.",
        notes="Do not mix CC0 structured data with BY-SA text from other namespaces.",
    ),
)


def _build_index(sources: Tuple[LexicalSource, ...]) -> Mapping[str, LexicalSource]:
    """Validate catalog uniqueness and return an immutable-by-convention index."""

    index: Dict[str, LexicalSource] = {}
    for source in sources:
        if source.source_id in index:
            raise ValueError(f"Duplicate lexical source id: {source.source_id}")
        index[source.source_id] = source
    return index


_SOURCE_BY_ID = _build_index(_SOURCES)


class SourceNotFoundError(KeyError):
    """Raised when a lexical source identifier is not registered."""


def iter_sources(*, unattended_only: bool = False) -> Iterator[LexicalSource]:
    """Yield registered sources in stable identifier order."""

    for source in sorted(_SOURCES, key=lambda item: item.source_id):
        if unattended_only and not source.unattended_eligible:
            continue
        yield source


def get_source(source_id: str) -> LexicalSource:
    """Return a registered source by its case-insensitive identifier."""

    normalized = source_id.strip().casefold()
    try:
        return _SOURCE_BY_ID[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_SOURCE_BY_ID))
        raise SourceNotFoundError(
            f"Unknown lexical source {source_id!r}. Available sources: {available}"
        ) from exc


def source_catalog_report(*, unattended_only: bool = False) -> Dict[str, object]:
    """Return the versioned catalog report used by the CLI and automation."""

    sources = list(iter_sources(unattended_only=unattended_only))
    return {
        "schema_version": SOURCE_CATALOG_VERSION,
        "notice": CATALOG_NOTICE,
        "filters": {"unattended_only": unattended_only},
        "count": len(sources),
        "sources": [source.to_dict() for source in sources],
    }


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
