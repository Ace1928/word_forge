"""Parser Refiner Module.

This module provides the core parsing and term refinement functionality for
Word Forge. It extracts lexical data from text, integrates with WordNet and
other lexical resources, and enriches word entries with relationships,
definitions, and usage examples.

Key Components:
    ParserRefiner: Main parsing pipeline that processes terms through the queue
    TermExtractor: NLP-based term discovery from text content
    ProcessingStatistics: Tracks processing metrics and statistics

The module uses NLTK for natural language processing, including lemmatization,
part-of-speech tagging, and WordNet integration. It supports concurrent
processing through a thread pool executor for efficient batch operations.

Architecture:
    Text Input → Term Extraction → Lemmatization → WordNet Lookup →
    Relationship Discovery → Database Storage → Queue Dispatch

Example:
    >>> parser = ParserRefiner(db_manager, queue_manager)
    >>> parser.process_word("algorithm")
    >>> stats = parser.get_stats()
"""

from __future__ import annotations

import logging  # Import logging
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple, cast

import nltk  # type: ignore
from nltk import Tree  # type: ignore
from nltk.corpus import wordnet as wn  # type: ignore
from nltk.corpus.reader.wordnet import Lemma as WordNetLemma  # type: ignore
from nltk.corpus.reader.wordnet import Synset as WordNetSynset
from nltk.stem import WordNetLemmatizer  # type: ignore

from word_forge.configs.config_essentials import LexicalDataset
from word_forge.database.database_manager import DBManager
from word_forge.parser.lexical_functions import WORDNET_LOCK, create_lexical_dataset
from word_forge.parser.linguistics import (
    canonicalize_language_tag,
    lookup_pronunciations,
    normalize_term,
    segment_graphemes,
)
from word_forge.queue.queue_manager import QueueManager
from word_forge.utils.nltk_utils import ensure_nltk_data

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from word_forge.parser.language_model import ModelState


@dataclass
class ProcessingStatistics:
    """Tracks and reports processing metrics with atomic counters."""

    processed_count: int = 0
    successful_count: int = 0
    error_count: int = 0

    def increment_processed(self) -> None:
        """Increment the processed counter."""
        self.processed_count += 1

    def increment_successful(self) -> None:
        """Increment the successful counter."""
        self.successful_count += 1

    def increment_error(self) -> None:
        """Increment the error counter."""
        self.error_count += 1

    def as_dict(self, queue_size: int, unique_words: int) -> Dict[str, int]:
        """Convert statistics to a dictionary including queue metrics."""
        return {
            "processed": self.processed_count,
            "successful": self.successful_count,
            "errors": self.error_count,
            "queue_size": queue_size,
            "unique_words": unique_words,
        }


@dataclass
class LexicalResources:
    """Manages resource paths for lexical data sources."""

    data_dir: str
    paths: Dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize resource paths based on data directory."""
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Define paths to all lexical resources
        self.paths = {
            "openthesaurus": f"{self.data_dir}/openthesaurus.jsonl",
            "odict": f"{self.data_dir}/odict.json",
            "dbnary": f"{self.data_dir}/dbnary.ttl",
            "opendict": f"{self.data_dir}/opendict.json",
            "thesaurus": f"{self.data_dir}/thesaurus.jsonl",
        }

    def get_path(self, resource_name: str) -> str:
        """Get the path for a specific resource."""
        return self.paths.get(resource_name, "")


class TermExtractor:
    """Discovers and extracts terms from textual content using advanced NLP techniques."""

    # Class-level flag to track whether NER is available
    _ner_available: bool = True

    _BCP47_TO_STOPWORDS = {
        "en": "english",
        "da": "danish",
        "nl": "dutch",
        "fi": "finnish",
        "fr": "french",
        "de": "german",
        "hu": "hungarian",
        "it": "italian",
        "no": "norwegian",
        "pt": "portuguese",
        "ro": "romanian",
        "ru": "russian",
        "es": "spanish",
        "sv": "swedish",
        "tr": "turkish",
    }

    def __init__(self) -> None:
        """Initialize the term extractor with necessary NLP components."""
        ensure_nltk_data()
        self._stop_words: FrozenSet[str] = frozenset(
            nltk.corpus.stopwords.words("english")  # type: ignore
        )
        self._common_words: FrozenSet[str] = frozenset(
            [
                "the",
                "and",
                "that",
                "have",
                "this",
                "with",
                "from",
                "they",
                "you",
                "what",
                "which",
                "their",
                "will",
                "would",
                "make",
                "when",
                "more",
                "other",
                "about",
                "some",
                "then",
                "than",
            ]
        )
        self._lemmatizer: WordNetLemmatizer = nltk.stem.WordNetLemmatizer()  # type: ignore

    @lru_cache(maxsize=1024)
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """
        Convert TreeBank POS tag to WordNet POS tag for accurate lemmatization.

        Args:
            treebank_tag: POS tag from NLTK's tagger

        Returns:
            WordNet POS constant for lemmatization
        """
        tag_map = {"J": wn.ADJ, "V": wn.VERB, "N": wn.NOUN, "R": wn.ADV}
        return cast(str, tag_map.get(treebank_tag[:1].upper(), wn.NOUN))

    def extract_terms(
        self, definition: str, examples: List[str], original_term: str, language: str = "en"
    ) -> Tuple[List[str], List[str]]:
        """
        Extract high-value lexical terms from definitions and examples.

        Args:
            definition: Consolidated definition text
            examples: List of usage examples
            original_term: The term being processed (to exclude from results)
            language: BCP 47 language of the definition text

        Returns:
            Tuple of (priority_terms, standard_terms) for processing
        """
        # Combine text with context markers to help NLP algorithms distinguish sources
        text_to_parse = f"DEFINITION: {definition} EXAMPLES: {' '.join(examples)}"
        original_term_lower = original_term.lower()

        # Initialize term collections
        discovered_terms: Set[str] = set()
        multiword_expressions: Set[str] = set()
        named_entities: Set[str] = set()

        # Fallback basic extraction (always performed for reliability)
        regex_terms = {
            word.lower() for word in re.findall(r"\b[a-zA-Z]{3,}\b", text_to_parse)
        }
        discovered_terms.update(regex_terms)

        primary_lang = language.split("-", 1)[0].lower()
        stopwords_lang = self._BCP47_TO_STOPWORDS.get(primary_lang, "english")
        try:
            stop_words = frozenset(nltk.corpus.stopwords.words(stopwords_lang))
        except Exception:
            stop_words = self._stop_words

        try:
            # Process text with advanced NLP techniques
            sentences: List[str] = nltk.sent_tokenize(text_to_parse)  # type: ignore
            for sentence in sentences:
                self._process_sentence(
                    sentence, discovered_terms, multiword_expressions, named_entities, stop_words
                )

            # Extract semantically related terms via WordNet
            semantic_terms = self._extract_semantic_terms(frozenset(discovered_terms))
            discovered_terms.update(semantic_terms)

        except Exception as e:
            # Fallback to regex-only results if NLP processing fails
            logger.warning(
                f"Advanced NLP processing failed, using regex fallback: {str(e)}",
                exc_info=True,
            )

        # Filter out problematic terms
        filtered_terms = self._filter_terms(
            discovered_terms, multiword_expressions, named_entities, original_term_lower
        )

        # Score and prioritize terms
        priority_terms, standard_terms = self._score_and_sort_terms(filtered_terms)

        return priority_terms, standard_terms

    def _process_sentence(
        self,
        sentence: str,
        discovered_terms: Set[str],
        multiword_expressions: Set[str],
        named_entities: Set[str],
        stop_words: FrozenSet[str],
    ) -> None:
        """
        Process a single sentence with multiple NLP techniques.

        Args:
            sentence: Text sentence to process
            discovered_terms: Set to collect individual terms
            multiword_expressions: Set to collect multiword expressions
            named_entities: Set to collect named entities
            stop_words: Active set of language-specific stopwords
        """
        # Basic tokenization and POS tagging
        tokens = nltk.word_tokenize(sentence)  # type: ignore
        tagged: List[Tuple[str, str]] = nltk.pos_tag(tokens)  # type: ignore

        # Extract single words with POS filtering and lemmatization
        for word, tag in tagged:
            word_lower: str = word.lower()  # Normalize to lowercase

            # Skip punctuation, short words, stop words, and numbers
            if (
                len(word_lower) < 3
                or not word_lower.isalpha()
                or word_lower in stop_words
            ):
                continue

            # Apply proper lemmatization based on part of speech
            wordnet_pos = self._get_wordnet_pos(tag)
            lemma = self._lemmatizer.lemmatize(word_lower, wordnet_pos)
            if len(lemma) >= 3:
                discovered_terms.add(lemma)

        # Named Entity Recognition for proper nouns and terms
        self._extract_named_entities(tagged, named_entities, discovered_terms, stop_words)

        # Detect useful multiword expressions
        self._extract_multiword_expressions(tagged, multiword_expressions)

    def _extract_named_entities(
        self,
        tagged: List[Tuple[str, str]],
        named_entities: Set[str],
        discovered_terms: Set[str],
        stop_words: FrozenSet[str],
    ) -> None:
        """
        Extract named entities from tagged tokens.

        Args:
            tagged: POS-tagged tokens
            named_entities: Set to add named entities to
            discovered_terms: Set to add component terms to
            stop_words: Active set of language-specific stopwords
        """
        # Skip NER if it previously failed (e.g., missing NLTK corpus data)
        if not TermExtractor._ner_available:
            return

        try:
            chunked: Tree = nltk.ne_chunk(tagged)  # type: ignore
            for subtree in chunked:
                if isinstance(subtree, Tree) and hasattr(subtree, "label"):
                    leaves: List[Tuple[str, str]] = cast(
                        List[Tuple[str, str]], subtree.leaves()
                    )
                    entity = " ".join(word for word, _ in leaves)
                    if len(entity) > 3:  # Filter out very short entities
                        entity_lower = entity.lower()
                        named_entities.add(entity_lower)
                        # Also add individual terms from the entity
                        for word in entity_lower.split():
                            lemma = self._lemmatizer.lemmatize(word)
                            if len(lemma) >= 3 and lemma not in stop_words:
                                discovered_terms.add(lemma)
        except Exception as e:
            # Soft fail for NER - disable for future calls and continue
            TermExtractor._ner_available = False
            logger.warning(
                f"Named entity recognition disabled due to error: {str(e)}. "
                "Processing will continue without NER."
            )

    def _extract_multiword_expressions(
        self, tagged: List[Tuple[str, str]], multiword_expressions: Set[str]
    ) -> None:
        """
        Extract multiword expressions using POS patterns.

        Args:
            tagged: POS-tagged tokens
            multiword_expressions: Set to collect found expressions
        """
        if len(tagged) < 2:
            return

        # Extract phrases based on common linguistic patterns
        for i in range(len(tagged) - 1):
            # Adjective + Noun pattern (e.g., "blue sky")
            if tagged[i][1].startswith("JJ") and tagged[i + 1][1].startswith("NN"):
                bigram = f"{tagged[i][0].lower()} {tagged[i+1][0].lower()}"
                if len(bigram) > 5:  # Avoid very short bigrams
                    multiword_expressions.add(bigram)

            # Noun + Noun pattern (e.g., "database system")
            if tagged[i][1].startswith("NN") and tagged[i + 1][1].startswith("NN"):
                bigram = f"{tagged[i][0].lower()} {tagged[i+1][0].lower()}"
                if len(bigram) > 5:  # Avoid very short bigrams
                    multiword_expressions.add(bigram)

            # Verb + Particle/Adverb pattern (e.g., "log in", "set up")
            if tagged[i][1].startswith("VB") and (
                tagged[i + 1][1] == "RP" or tagged[i + 1][1].startswith("RB")
            ):
                bigram = f"{tagged[i][0].lower()} {tagged[i+1][0].lower()}"
                if len(bigram) > 5:  # Avoid very short bigrams
                    multiword_expressions.add(bigram)

    @lru_cache(maxsize=128)
    def _extract_semantic_terms(self, base_terms: FrozenSet[str]) -> Set[str]:
        """
        Find semantically related terms through WordNet.

        This function discovers and extracts semantically related terms by traversing
        WordNet's lexical database. It explores three relationship types to build a
        comprehensive semantic network:

        1. Synonyms - Words with the same meaning
        2. Hypernyms - Broader category terms (e.g., 'vehicle' is a hypernym of 'car')
        3. Hyponyms - More specific terms (e.g., 'sedan' is a hyponym of 'car')

        Args:
            base_terms: Initial set of discovered terms to find semantic relations for

        Returns:
            Set[str]: Collection of semantically related terms, limited to 200 results

        Note:
            The function implements performance optimizations:
            - Processes only a subset of input terms (max 75) to prevent combinatorial explosion
            - Uses LRU caching for repeated invocations with identical inputs
            - Limits result set size to 200 terms to prevent downstream overload
            - Silently continues on WordNet lookup failures (term not found, etc.)
        """
        ensure_nltk_data()
        semantic_terms: Set[str] = set()
        term_sample = sorted(base_terms, key=normalize_term)[:75]

        def _process_lemma(lemma: WordNetLemma) -> None:
            """Extract and normalize a lemma name, adding to results if valid."""
            lemma_name = lemma.name()
            if not isinstance(lemma_name, str):
                return

            # Normalize: replace underscores with spaces, convert to lowercase
            processed_name = lemma_name.replace("_", " ").lower()

            # Only include terms meeting minimum length requirement
            if len(processed_name) >= 3 and processed_name.replace(" ", "").isalpha():
                semantic_terms.add(processed_name)

        # Process each base term
        with WORDNET_LOCK:
            for base_term in term_sample:
                try:
                    # Retrieve all synsets (word senses) for the term
                    synsets: List[WordNetSynset] = wn.synsets(base_term)  # type: ignore
                    for synset in synsets:
                        # 1. Process direct synonyms from the synset
                        for lemma in synset.lemmas():  # type: ignore
                            _process_lemma(lemma)  # type: ignore

                        # 2. Process hypernyms (broader/parent categories)
                        for hypernym in synset.hypernyms():  # type: ignore
                            for lemma in hypernym.lemmas():  # type: ignore
                                _process_lemma(lemma)  # type: ignore

                        # 3. Process hyponyms (more specific/child categories)
                        for hyponym in synset.hyponyms():  # type: ignore
                            for lemma in hyponym.lemmas():  # type: ignore
                                _process_lemma(lemma)  # type: ignore

                except (LookupError, AttributeError, ValueError, TypeError):
                    continue
                except Exception as unexpected_e:
                    logger.warning(
                        f"Unexpected error during WordNet lookup for '{base_term}': {unexpected_e}",
                        exc_info=True,
                    )
                    continue

        # Limit returned set to prevent overwhelming downstream processing
        return set(sorted(semantic_terms, key=normalize_term)[:200])

    def _filter_terms(
        self,
        discovered_terms: Set[str],
        multiword_expressions: Set[str],
        named_entities: Set[str],
        original_term: str,
    ) -> Set[str]:
        """
        Filter the collected terms to remove unwanted items.

        Args:
            discovered_terms: All collected terms
            multiword_expressions: Multiword expressions to preserve
            named_entities: Named entities to preserve
            original_term: Original term being processed (to exclude)

        Returns:
            Filtered set of terms
        """
        # Create combined set with all terms
        all_terms = discovered_terms.union(multiword_expressions, named_entities)

        # Remove the original term
        all_terms.discard(original_term)

        # Remove very common function words
        all_terms -= self._common_words

        return all_terms

    def _score_and_sort_terms(self, terms: Set[str]) -> Tuple[List[str], List[str]]:
        """
        Score and sort terms by potential lexical value.

        Args:
            terms: Set of all filtered terms

        Returns:
            Tuple of (priority_terms, standard_terms)
        """
        scored_terms: List[Tuple[str, int]] = []

        for term in terms:
            # Scoring heuristics: length, complexity, multiword bonus
            score = len(term)  # Base score is length
            score += term.count(" ") * 3  # Multiword expressions get a boost
            score += sum(
                1 for c in term if c not in "aeiou"
            )  # Consonant density correlates with specificity
            scored_terms.append((term, score))

        # Sort by score (higher is better)
        sorted_terms = sorted(
            scored_terms,
            key=lambda item: (-item[1], normalize_term(item[0])),
        )

        # Split into two categories: priority and standard
        priority_terms = [term for term, _ in sorted_terms if " " in term][:100]
        other_terms = [term for term, _ in sorted_terms if " " not in term][:150]

        return priority_terms, other_terms


class ParserRefiner:
    """
    Uses advanced lexical lookups to:
    1. Retrieve definitions, synonyms, antonyms, usage examples, etc.
    2. Store them in the DB.
    3. Enqueue new words discovered from that data for further processing.
    """

    def __init__(
        self,
        db_manager: Optional[DBManager] = None,
        queue_manager: Optional[QueueManager[str]] = None,
        data_dir: str = "data",
        model_name: Optional[str] = None,
        model_profile: Optional[str] = None,
        llm_state: Optional[ModelState] = None,
        language: Optional[str] = None,
    ) -> None:
        """
        Initialize the ParserRefiner with database and queue managers.

        Args:
            db_manager: DBManager instance for database operations
            queue_manager: QueueManager instance for enqueuing new terms
            data_dir: Path to the folder containing lexical resources
            model_name: Optional custom Hugging Face model identifier
            model_profile: Named model profile, including ``auto`` and ``off``
            llm_state: Preconfigured model state, primarily for embedded use
            language: BCP 47 language for this ingestion queue
        """
        from word_forge.config import config

        self.db_manager = db_manager or DBManager()
        self.queue_manager = (
            queue_manager if queue_manager is not None else QueueManager[str]()
        )
        self.resources = LexicalResources(data_dir)
        self.term_extractor = TermExtractor()
        self.stats = ProcessingStatistics()
        self.language = canonicalize_language_tag(
            language or config.parser.default_language
        )
        self._reported_source_warnings: Set[str] = set()
        self._warning_lock = Lock()

        self.llm_state = llm_state
        selected_model: Optional[str] = None
        if self.llm_state is None and model_name:
            selected_model = model_name
        elif self.llm_state is None and model_profile:
            from word_forge.parser.model_profiles import resolve_model_profile

            selected_profile = resolve_model_profile(model_profile)
            selected_model = selected_profile.model_id
        elif self.llm_state is None:
            from word_forge.parser.model_profiles import resolve_model_profile

            if config.parser.enable_model:
                selected_model = config.parser.model_name
                if selected_model is None:
                    selected_model = resolve_model_profile(
                        config.parser.model_profile
                    ).model_id
        if self.llm_state is None and selected_model is not None:
            from word_forge.parser.language_model import ModelState

            self.llm_state = ModelState(selected_model)

        # Initialize thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=5)

    def process_word(self, term: str) -> bool:
        """
        Process a word using the integrated lexical resources.

        Data is stored in the DB and new terms are enqueued for further processing.

        Args:
            term: The word to process

        Returns:
            Boolean indicating whether processing was successful
        """
        if not isinstance(term, str) or not term.strip():
            return False
        display_term = unicodedata.normalize("NFC", term.strip())

        try:
            self.stats.increment_processed()

            # Retrieve comprehensive lexical data
            dataset = create_lexical_dataset(
                display_term,
                openthesaurus_path=self.resources.get_path("openthesaurus"),
                odict_path=self.resources.get_path("odict"),
                dbnary_path=self.resources.get_path("dbnary"),
                opendict_path=self.resources.get_path("opendict"),
                thesaurus_path=self.resources.get_path("thesaurus"),
                model_state=self.llm_state,
                language=self.language,
            )
            self._report_source_warnings(dataset.get("source_warnings", []))

            # Extract and consolidate word information
            definitions = self._extract_all_definitions(dataset)
            full_definition = " | ".join(definitions) if definitions else ""
            part_of_speech = self._extract_part_of_speech(dataset)
            usage_examples = self._extract_usage_examples(dataset)

            # Store in database
            has_lexical_data = bool(
                definitions
                or part_of_speech
                or usage_examples
                or dataset.get("wordnet_data")
                or dataset.get("dbnary_data")
                or dataset.get("openthesaurus_synonyms")
                or dataset.get("thesaurus_synonyms")
            )
            word_id = self.db_manager.insert_or_update_word(
                term=display_term,
                definition=full_definition,
                part_of_speech=part_of_speech,
                usage_examples=usage_examples,
                language=self.language,
                source=self._primary_source(dataset),
                is_stub=not has_lexical_data,
            )
            self.db_manager.replace_graphemes(word_id, segment_graphemes(display_term))
            self.db_manager.replace_pronunciations(
                word_id, lookup_pronunciations(display_term, self.language)
            )

            # Relationship persistence and discovery may run in parallel, but
            # both must complete before this term is reported as successful.
            # Returning earlier lets callers stop the queue while these tasks
            # are still trying to enqueue newly discovered terms.
            relationship_future = self._executor.submit(
                self._process_relationships, display_term, dataset
            )
            discovery_future = self._executor.submit(
                self._discover_new_terms,
                display_term,
                full_definition,
                usage_examples,
            )
            relationship_future.result()
            discovery_future.result()

            self.stats.increment_successful()
            return True

        except Exception as e:
            self.stats.increment_error()
            logger.error(
                "Error processing word '%s' (%s): %s",
                display_term,
                self.language,
                str(e),
                exc_info=True,
            )
            return False

    def _report_source_warnings(self, warnings: List[str]) -> None:
        """Log each optional-source warning once per parser instance."""

        with self._warning_lock:
            for warning in warnings:
                if warning in self._reported_source_warnings:
                    continue
                self._reported_source_warnings.add(warning)
                logger.warning("Lexical source unavailable: %s", warning)

    def _primary_source(self, dataset: LexicalDataset) -> str:
        """Return the highest-priority source that contributed lexical data."""

        wordnet_data = dataset.get("wordnet_data", [])
        if wordnet_data:
            return str(wordnet_data[0].get("source") or "princeton-wordnet")
        if dataset.get("dbnary_data"):
            return "dbnary"
        if dataset.get("odict_data", {}).get("definition") not in {"", "Not Found"}:
            return "local-odict"
        if dataset.get("opendict_data", {}).get("definition") not in {
            "",
            "Not Found",
        }:
            return "local-opendict"
        if dataset.get("openthesaurus_synonyms"):
            return "local-openthesaurus"
        if dataset.get("thesaurus_synonyms"):
            return "local-thesaurus"
        return "user-seed"

    @staticmethod
    def _extract_all_definitions(dataset: LexicalDataset) -> List[str]:
        """
        Extract and deduplicate definitions from all sources.

        Args:
            dataset: Comprehensive lexical dataset for a word

        Returns:
            List of unique definitions from all sources
        """
        combined_definitions: List[str] = []
        seen_definitions: Set[str] = set()
        target_primary = dataset["language"].split("-", 1)[0]

        def append_definition(value: object) -> None:
            """Append a non-placeholder definition once, preserving source order."""

            if not isinstance(value, str):
                return
            definition = value.strip()
            if (
                not definition
                or definition == "Not Found"
                or definition in seen_definitions
            ):
                return
            seen_definitions.add(definition)
            combined_definitions.append(definition)

        # WordNet definitions
        # Use .get with default empty list
        for wn_data in dataset.get("wordnet_data", []):
            # Ensure wn_data is a dict before accessing keys
            if (
                isinstance(wn_data, dict)
                and str(wn_data.get("definition_language", "en")).split("-", 1)[0]
                == target_primary
            ):
                append_definition(wn_data.get("definition", ""))

        # ODict / OpenDictData
        odict_data = dataset.get("odict_data", {})
        # Add type check for odict_data
        odict_def: Optional[str] = (
            odict_data.get("definition", "") if isinstance(odict_data, dict) else None
        )
        append_definition(odict_def)

        opendict_data = dataset.get("opendict_data", {})
        # Add type check for opendict_data
        open_dict_def: Optional[str] = (
            opendict_data.get("definition", "")
            if isinstance(opendict_data, dict)
            else None
        )
        append_definition(open_dict_def)

        # Dbnary definitions
        # Use .get with default empty list
        for item in dataset.get("dbnary_data", []):
            # Ensure item is a dict before accessing keys
            if isinstance(item, dict):
                definition_language = str(item.get("definition_language", ""))
                if (
                    not definition_language
                    or definition_language.split("-", 1)[0] == target_primary
                ):
                    append_definition(item.get("definition", ""))

        return combined_definitions

    def _extract_part_of_speech(self, dataset: LexicalDataset) -> str:
        """
        Extract part of speech from WordNet data if available.

        Args:
            dataset: Comprehensive lexical dataset for a word

        Returns:
            Part of speech string, or empty string if not available
        """
        # Use .get with default empty list
        wordnet_data = dataset.get("wordnet_data", [])
        # Ensure list is not empty and first element is a dict
        if (
            wordnet_data
            and isinstance(wordnet_data, list)
            and isinstance(wordnet_data[0], dict)
        ):
            pos: Optional[str] = wordnet_data[0].get("part_of_speech", "")
            return pos if pos else ""  # Return empty string if None or empty
        return ""

    def _extract_usage_examples(self, dataset: LexicalDataset) -> List[str]:
        """
        Extract usage examples from all sources.

        Args:
            dataset: Comprehensive lexical dataset for a word

        Returns:
            List of unique usage examples from all sources
        """
        usage_examples: List[str] = []
        seen_examples: Set[str] = set()
        target_primary = dataset["language"].split("-", 1)[0]

        def append_example(value: object) -> None:
            """Append one non-placeholder example in stable source order."""

            if not isinstance(value, str):
                return
            example = value.strip()
            if (
                not example
                or "No example available" in example
                or example in seen_examples
            ):
                return
            seen_examples.add(example)
            usage_examples.append(example)

        # WordNet examples
        for wn_data in dataset.get("wordnet_data", []):  # Use .get
            if (
                wn_data.get("examples_language", "en").split("-", 1)[0]
                != target_primary
            ):
                continue
            for ex in wn_data.get("examples", []):
                append_example(ex)

        # Add auto-generated example sentence
        auto_ex = dataset.get("example_sentence", "")  # Use .get
        append_example(auto_ex)

        return usage_examples

    def _process_relationships(self, term: str, dataset: LexicalDataset) -> None:
        """
        Process and store word relationships.

        Args:
            term: The base term
            dataset: Comprehensive lexical dataset for the term
        """
        relationship_cache: Set[Tuple[str, str, str, str]] = set()
        discovered_terms: Dict[str, str] = {}
        base_language = dataset["language"]

        def record_relationship(
            related_term: object,
            relationship_type: str,
            source: str,
            related_language: str = base_language,
        ) -> None:
            """Persist one normalized assertion and queue same-language targets."""

            if not isinstance(related_term, str) or not related_term.strip():
                return
            display_related = unicodedata.normalize("NFC", related_term.strip())
            try:
                canonical_related_language = canonicalize_language_tag(related_language)
            except ValueError:
                logger.warning(
                    "Skipping %s relationship with invalid language %r",
                    source,
                    related_language,
                )
                return
            normalized_related = normalize_term(display_related)
            if (
                normalized_related == normalize_term(term)
                and canonical_related_language == base_language
            ):
                return
            relationship_key = (
                normalized_related,
                canonical_related_language,
                relationship_type,
                source,
            )
            if relationship_key in relationship_cache:
                return
            self.db_manager.insert_relationship(
                term,
                display_related,
                relationship_type,
                base_language=base_language,
                related_language=canonical_related_language,
                source=source,
            )
            relationship_cache.add(relationship_key)
            if canonical_related_language == base_language:
                discovered_terms.setdefault(normalized_related, display_related)

        # Process WordNet relationships
        for wn_data in dataset.get("wordnet_data", []):
            for syn in wn_data.get("synonyms", []):
                record_relationship(syn, "synonym", wn_data["source"])
            for ant in wn_data.get("antonyms", []):
                record_relationship(ant, "antonym", wn_data["source"])

            # Extract multilingual OMW translations from Princeton WordNet
            synset_id = wn_data.get("synset_id")
            if synset_id:
                try:
                    from nltk.corpus import wordnet as wn
                    from word_forge.parser.wordnet_languages import _PRIMARY_TO_WORDNET
                    
                    synset = wn.synset(synset_id)
                    for primary_lang, nltk_lang in _PRIMARY_TO_WORDNET.items():
                        if primary_lang == base_language.split("-", 1)[0]:
                            continue
                        omw_lemmas = synset.lemma_names(lang=nltk_lang)
                        for lemma in omw_lemmas:
                            record_relationship(
                                lemma.replace("_", " "),
                                "translation",
                                "nltk-omw",
                                related_language=primary_lang,
                            )
                except Exception:
                    pass

        # OpenThesaurus synonyms
        for s in dataset.get("openthesaurus_synonyms", []):
            record_relationship(s, "synonym", "local-openthesaurus")

        # Thesaurus synonyms
        for s in dataset.get("thesaurus_synonyms", []):
            record_relationship(s, "synonym", "local-thesaurus")

        # Translations from DBnary
        for item in dataset.get("dbnary_data", []):
            translation = item.get("translation", "")
            translation_language = item.get("translation_language", "")
            if translation and translation_language:
                record_relationship(
                    translation,
                    "translation",
                    "dbnary",
                    translation_language,
                )

        # Batch enqueue all discovered terms
        for discovered_term in sorted(discovered_terms.values(), key=normalize_term):
            self.queue_manager.enqueue(discovered_term)

    def _discover_new_terms(
        self, term: str, definition: str, examples: List[str]
    ) -> None:
        """
        Discover and enqueue new terms from definitions and examples using advanced NLP techniques.

        Args:
            term: The base term being processed
            definition: The term's consolidated definition
            examples: List of usage examples for the term
        """
        priority_terms, standard_terms = self.term_extractor.extract_terms(
            definition, examples, term, language=self.language
        )

        # Enqueue priority terms first (multiword expressions and specialized terms)
        for new_term in priority_terms:
            if normalize_term(new_term) != normalize_term(term):
                self.queue_manager.enqueue(new_term)

        # Then enqueue other discovered terms
        for new_term in standard_terms:
            if normalize_term(new_term) != normalize_term(term):
                self.queue_manager.enqueue(new_term)

    def get_stats(self) -> Dict[str, int]:
        """
        Get processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        queue_size = self.queue_manager.size if self.queue_manager else 0
        unique_words = 0
        if self.queue_manager and hasattr(self.queue_manager, "_seen_items"):
            try:
                unique_words = len(self.queue_manager._seen_items)  # type: ignore
            except TypeError:
                pass

        return self.stats.as_dict(
            queue_size=queue_size,
            unique_words=unique_words,
        )

    def shutdown(self) -> None:
        """Gracefully shut down resources like thread pools."""
        self._executor.shutdown(wait=True)


# Export all components for module usage
__all__ = ["ParserRefiner", "TermExtractor", "LexicalResources", "ProcessingStatistics"]
