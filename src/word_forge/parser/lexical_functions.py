# filepath: /home/lloyd/eidosian_forge/word_forge/src/word_forge/parser/lexical_functions.py
# ============================================================================
#                              IMPORTS
# ============================================================================
from __future__ import annotations

import functools
import json
import os
import re
import typing
from contextlib import contextmanager
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from nltk.corpus import wordnet as wn  # type: ignore
from nltk.corpus.reader.wordnet import Lemma, Synset  # type: ignore

# Optional dependency for DBnary RDF processing
try:
    from rdflib import Graph
    from rdflib import Literal as RdfLiteral
    from rdflib.query import ResultRow

    _rdflib_available = True
except ImportError:
    _rdflib_available = False
    Graph = None  # type: ignore[assignment]
    RdfLiteral = None  # type: ignore[assignment]
    ResultRow = None  # type: ignore[assignment]

from word_forge.configs.config_essentials import (
    DbnaryEntry,
    DictionaryEntry,
    JsonData,
    LexicalDataset,
    LexicalResourceError,
    ResourceParsingError,
    T,
    WordnetEntry,
)
from word_forge.parser.linguistics import canonicalize_language_tag
from word_forge.parser.wordnet_languages import (
    WordNetLanguageError,
    resolve_wordnet_language,
)
from word_forge.utils.nltk_utils import ensure_nltk_data

if TYPE_CHECKING:
    from word_forge.parser.language_model import ModelState


# ============================================================================
#                           FILE OPERATIONS
# ============================================================================
def file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if a file exists at the specified path.

    Args:
        file_path: Path to check for file existence

    Returns:
        True if the file exists, False otherwise
    """
    return os.path.isfile(file_path)


@contextmanager
def safe_open(
    file_path: Union[str, Path], mode: str = "r", encoding: str = "utf-8"
) -> Iterator[Optional[IO[Any]]]:
    """
    Safely open a file, handling non-existent files and IO errors.

    Args:
        file_path: Path to the file to open
        mode: File mode (r, w, etc.)
        encoding: Text encoding to use

    Yields:
        File handle if file exists and can be opened, None otherwise

    Raises:
        LexicalResourceError: If file exists but cannot be opened due to IO errors
    """
    if not file_exists(file_path):
        yield None
        return

    try:
        with open(file_path, mode, encoding=encoding) as f:
            fh: IO[Any] = f
            yield fh
    except (IOError, OSError) as e:
        raise LexicalResourceError(f"Error opening file {file_path}: {str(e)}")


def read_json_file(
    file_path: Union[str, Path], default_value: T = None
) -> Union[JsonData, T]:
    """
    Read and parse a JSON file, returning a default value if the file doesn't exist or is invalid.

    Args:
        file_path: Path to the JSON file
        default_value: Value to return if file doesn't exist or is invalid

    Returns:
        Parsed JSON data or the default value

    Raises:
        LexicalResourceError: If file exists but cannot be opened
    """
    with safe_open(file_path) as fh:
        if fh is None:
            return default_value
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return default_value


def read_jsonl_file(
    file_path: Union[str, Path], process_func: Callable[[Dict[str, Any]], Optional[T]]
) -> List[T]:
    """
    Read and process a JSON Lines file line by line.

    Args:
        file_path: Path to the JSONL file
        process_func: Function to process each parsed JSON line

    Returns:
        List of processed results

    Raises:
        LexicalResourceError: If file cannot be accessed or processing fails
    """
    results: List[T] = []
    with safe_open(file_path) as fh:
        if fh is None:
            return results

        line_num = 0
        try:
            for line in fh:
                line_num += 1
                if not line.strip():
                    continue

                data = json.loads(line)
                processed = process_func(data)
                if processed is not None:
                    results.append(processed)
        except Exception as e:
            raise ResourceParsingError(
                f"Error processing line {line_num} in {file_path}: {str(e)}"
            )

    return results


# ============================================================================
#                            WORDNET FUNCTIONS
# ============================================================================
def get_synsets(word: str, language: str = "en") -> List[Synset]:
    """Retrieve synsets using an explicit BCP 47 lexical language."""

    ensure_nltk_data()
    resolved = resolve_wordnet_language(language)
    return list(_get_synsets_cached(word, resolved.nltk_code))


@functools.lru_cache(maxsize=4096)
def _get_synsets_cached(word: str, nltk_language: str) -> Tuple[Synset, ...]:
    """Cache immutable NLTK lookup results by term and ISO 639-3 code."""

    return tuple(wn.synsets(word, lang=nltk_language))  # type: ignore[no-any-return]


def get_wordnet_data(word: str, language: str = "en") -> List[WordnetEntry]:
    """
    Extract comprehensive linguistic data from WordNet for a given word.

    Args:
        word: Word to retrieve data for
        language: BCP 47 language tag for the lemma lookup

    Returns:
        List of structured entries containing definitions, examples, synonyms, antonyms,
        and part-of-speech information
    """
    results: List[WordnetEntry] = []
    resolved = resolve_wordnet_language(language)
    synsets: List[Synset] = get_synsets(word, resolved.bcp47)

    for synset in synsets:
        lemmas: List[Lemma] = synset.lemmas(lang=resolved.nltk_code) or []
        synonyms: List[str] = []
        for lemma in lemmas:
            name = lemma.name()
            if isinstance(name, str):
                synonyms.append(name.replace("_", " "))

        # Extract antonyms from lemmas
        antonyms: List[str] = []
        for lemma in lemmas:
            lemma_antonyms: List[Lemma] = lemma.antonyms()
            for antonym in lemma_antonyms:
                if antonym.lang() != resolved.nltk_code:
                    continue
                antonym_name = antonym.name()
                if isinstance(antonym_name, str):
                    antonym_name = antonym_name.replace("_", " ")
                    antonyms.append(antonym_name)

        # Explicitly type with expected return types but handle variations
        # Cast the result to Optional[str] as nltk types might be incomplete
        definition_result: Optional[str] = typing.cast(
            Optional[str], synset.definition()
        )
        definition: str = definition_result if definition_result is not None else ""

        # Cast the result to Optional[List[str]]
        examples_result: Optional[List[str]] = typing.cast(
            Optional[List[str]], synset.examples()
        )
        examples: List[str] = examples_result if examples_result is not None else []

        # Cast the result to Optional[str]
        pos_result: Optional[str] = typing.cast(Optional[str], synset.pos())
        pos: str = pos_result if pos_result is not None else ""

        results.append(
            WordnetEntry(
                word=word,
                language=resolved.bcp47,
                source=resolved.source_id,
                synset_id=str(synset.name()),
                definition=definition,
                definition_language="en",
                examples=examples,
                examples_language="en",
                synonyms=synonyms,
                antonyms=antonyms,
                part_of_speech=pos,
            )
        )

    return results


# ============================================================================
#                          LEXICAL DATA SOURCES
# ============================================================================
def get_openthesaurus_data(word: str, openthesaurus_path: str) -> List[str]:
    """
    Extract synonyms from OpenThesaurus for a given word.

    Args:
        word: Word to retrieve synonyms for
        openthesaurus_path: Path to the OpenThesaurus JSONL file

    Returns:
        List of unique synonyms with duplicates removed while preserving order
    """

    def process_line(data: Dict[str, Any]) -> Optional[List[str]]:
        words = data.get("words", [])
        if word in words:
            return [w for w in words if w != word]
        return None

    synonyms: List[str] = []
    for syns in read_jsonl_file(openthesaurus_path, process_line):
        synonyms.extend(syns)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(synonyms))


def get_odict_data(word: str, odict_path: str) -> DictionaryEntry:
    """
    Retrieve dictionary data from ODict for a given word.

    Args:
        word: Word to retrieve data for
        odict_path: Path to the ODict JSON file

    Returns:
        Dictionary containing definition and usage examples
    """
    default_entry: DictionaryEntry = {
        "definition": "Not Found",
        "examples": [],
    }
    odict_data = read_json_file(odict_path, {})
    entry = (
        odict_data.get(word, default_entry)
        if isinstance(odict_data, dict)
        else default_entry
    )
    if isinstance(entry, dict) and "definition" in entry and "examples" in entry:
        examples_raw = entry.get("examples", [])
        if isinstance(examples_raw, list):
            examples = [str(ex) for ex in examples_raw]
        else:
            examples = []
        return DictionaryEntry(definition=str(entry["definition"]), examples=examples)
    return default_entry


def get_dbnary_data(
    word: str, dbnary_path: str, language: str = "en"
) -> List[DbnaryEntry]:
    """
    Extract linguistic data from DBnary RDF for a given word.

    Args:
        word: Word to retrieve data for
        dbnary_path: Path to the DBnary TTL file
        language: BCP 47 language of the written form

    Returns:
        List of entries containing definitions and translations

    Raises:
        LexicalResourceError: If there's an error processing the DBnary data
    """
    if not _rdflib_available:
        return []

    if not file_exists(dbnary_path):
        return []

    try:
        canonical_language = canonicalize_language_tag(language)
        graph = Graph()
        graph.parse(dbnary_path, format="ttl")

        sparql_query = """
        PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?definition ?translation
        WHERE {
          ?entry ontolex:canonicalForm/ontolex:writtenRep ?written .
          FILTER (?written = ?lookup)
          OPTIONAL { ?entry ontolex:definition/rdfs:label ?definition . }
          OPTIONAL { ?entry ontolex:translation/rdfs:label ?translation . }
        }
        """

        results = graph.query(
            sparql_query,
            initBindings={"lookup": RdfLiteral(word, lang=canonical_language)},
        )
        output: List[DbnaryEntry] = []

        for row in results:
            # Ensure row is a ResultRow before accessing elements by index/key
            if not isinstance(row, ResultRow):
                continue

            # Access elements safely using .get() or check length
            definition_node = row.definition if hasattr(row, "definition") else None
            translation_node = row.translation if hasattr(row, "translation") else None

            # Convert rdflib Nodes (Literal, URIRef) to string safely
            definition = (
                str(definition_node.value)
                if isinstance(definition_node, RdfLiteral)
                else ""
            )
            translation = (
                str(translation_node.value)
                if isinstance(translation_node, RdfLiteral)
                else ""
            )

            if definition or translation:
                output.append(
                    DbnaryEntry(
                        definition=definition,
                        definition_language=(
                            str(definition_node.language or "")
                            if isinstance(definition_node, RdfLiteral)
                            else ""
                        ),
                        translation=translation,
                        translation_language=(
                            str(translation_node.language or "")
                            if isinstance(translation_node, RdfLiteral)
                            else ""
                        ),
                    )
                )

        return output

    except Exception as e:
        raise LexicalResourceError(f"Error processing Dbnary data: {str(e)}")


def get_opendictdata(word: str, opendict_path: str) -> DictionaryEntry:
    """
    Retrieve dictionary data from OpenDict for a given word.

    Args:
        word: Word to retrieve data for
        opendict_path: Path to the OpenDict JSON file

    Returns:
        Dictionary containing definition and examples
    """
    default_entry: DictionaryEntry = {
        "definition": "Not Found",
        "examples": [],
    }
    data = read_json_file(opendict_path, {})
    entry = data.get(word, default_entry) if isinstance(data, dict) else default_entry
    if isinstance(entry, dict) and "definition" in entry and "examples" in entry:
        # Ensure examples are processed correctly into List[str]
        examples_raw = entry.get("examples", [])
        examples: List[str] = []
        if isinstance(examples_raw, list):
            examples = [
                str(ex) for ex in examples_raw if isinstance(ex, (str, int, float))
            ]
        return DictionaryEntry(definition=str(entry["definition"]), examples=examples)
    return default_entry


def get_thesaurus_data(word: str, thesaurus_path: str) -> List[str]:
    """
    Extract synonyms from Thesaurus for a given word.

    Args:
        word: Word to retrieve synonyms for
        thesaurus_path: Path to the Thesaurus JSONL file

    Returns:
        List of synonyms from the thesaurus source
    """

    def process_line(data: Dict[str, Any]) -> Optional[List[str]]:
        if word == data.get("word"):
            synonyms = data.get("synonyms", [])
            if isinstance(synonyms, list):
                return [item for item in synonyms if isinstance(item, str)]
        return None

    results: List[str] = []
    for syns in read_jsonl_file(thesaurus_path, process_line):
        results.extend(syns)

    return results


# ============================================================================
#                       EXAMPLE GENERATION FUNCTIONS
# ============================================================================
def generate_example_usage(
    word: str,
    definition: str,
    synonyms: List[str],
    antonyms: List[str],
    pos: str,
    model_state: ModelState,
    language: str = "en",
) -> str:
    """
    Generate an example sentence for a word using a language model.

    Args:
        word: The target word to use in the example
        definition: The word's definition
        synonyms: List of word synonyms
        antonyms: List of word antonyms
        pos: Part of speech
        model_state: Initialized :class:`ModelState` instance to use for text generation

    Returns:
        A generated example sentence or an error message
    """
    # Construct prompt with word details
    prompt = (
        f"Word: {word}\n"
        f"Part of Speech: {pos}\n"
        f"Definition: {definition}\n"
        f"Synonyms: {', '.join(synonyms[:5])}\n"
        f"Antonyms: {', '.join(antonyms[:3])}\n"
        f"Language: {canonicalize_language_tag(language)}\n"
        f"Task: Generate a single concise example sentence using the word '{word}'.\n"
        f"Example Sentence: "
    )

    # Use the provided model state for generation
    generated_text = model_state.generate_text(prompt)
    full_text = str(generated_text) if generated_text else ""

    if not full_text:
        return f"Could not generate example for '{word}'."

    # Parse out just the generated example
    if "Example Sentence:" in full_text:
        parts = full_text.split("Example Sentence:")
        if len(parts) > 1:
            example = parts[1].strip()
            # Capture up to first period for a complete sentence
            if "." in example:
                sentence_end = example.find(".") + 1
                return example[:sentence_end].strip()
            return example

    # If we got text but couldn't parse it properly, return it as-is
    if full_text and not full_text.startswith("Could not"):
        # Try to find the first complete sentence
        sentences = re.split(r"[.!?]", full_text)
        if sentences and len(sentences[0]) > 5:  # Minimum length for a valid sentence
            return sentences[0].strip() + "."
        return full_text.strip()

    return f"Could not extract valid example for '{word}'."


# ============================================================================
#                          DATASET CREATION
# ============================================================================
def create_lexical_dataset(
    word: str,
    openthesaurus_path: str = "data/openthesaurus.jsonl",
    odict_path: str = "data/odict.json",
    dbnary_path: str = "data/dbnary.ttl",
    opendict_path: str = "data/opendict.json",
    thesaurus_path: str = "data/thesaurus.jsonl",
    model_state: Optional[ModelState] = None,
    language: str = "en",
) -> LexicalDataset:
    """
    Create a comprehensive dataset of lexical information for a word.

    Args:
        word: The word to gather data for
        openthesaurus_path: Path to OpenThesaurus data
        odict_path: Path to ODict data
        dbnary_path: Path to DBnary data
        opendict_path: Path to OpenDict data
        thesaurus_path: Path to Thesaurus data
        model_state: Optional :class:`ModelState` used to generate example sentences
        language: BCP 47 language tag for the lexical item

    Returns:
        Dictionary containing comprehensive lexical data from all sources
    """
    canonical_language = canonicalize_language_tag(language)
    source_warnings: List[str] = []
    try:
        wordnet_data = get_wordnet_data(word, canonical_language)
    except WordNetLanguageError as exc:
        wordnet_data = []
        source_warnings.append(str(exc))

    dataset: LexicalDataset = {
        "word": word,
        "language": canonical_language,
        "wordnet_data": wordnet_data,
        "openthesaurus_synonyms": get_openthesaurus_data(word, openthesaurus_path),
        "odict_data": get_odict_data(word, odict_path),
        "dbnary_data": get_dbnary_data(word, dbnary_path, canonical_language),
        "opendict_data": get_opendictdata(word, opendict_path),
        "thesaurus_synonyms": get_thesaurus_data(word, thesaurus_path),
        "example_sentence": "",
        "source_warnings": source_warnings,
    }

    # Prefer source-authored examples. Generative enrichment only fills a gap,
    # which keeps the default pipeline offline, fast, and reproducible.
    if wordnet_data:
        source_example = next(
            (
                example.strip()
                for entry in wordnet_data
                for example in entry.get("examples", [])
                if example.strip()
                and entry.get("examples_language", "en").split("-", 1)[0]
                == canonical_language.split("-", 1)[0]
            ),
            "",
        )
        if source_example:
            dataset["example_sentence"] = source_example
        elif model_state is not None:
            first_entry = wordnet_data[0]
            dataset["example_sentence"] = generate_example_usage(
                word,
                definition=first_entry.get("definition", ""),
                synonyms=dataset["openthesaurus_synonyms"],
                antonyms=first_entry.get("antonyms", []),
                pos=first_entry.get("part_of_speech", ""),
                model_state=model_state,
                language=canonical_language,
            )

    return dataset


# ============================================================================
#                                 EXPORTS
# ============================================================================
__all__ = [
    # File operations
    "file_exists",
    "safe_open",
    "read_json_file",
    "read_jsonl_file",
    # WordNet functions
    "get_synsets",
    "get_wordnet_data",
    # Lexical data sources
    "get_openthesaurus_data",
    "get_odict_data",
    "get_dbnary_data",
    "get_opendictdata",
    "get_thesaurus_data",
    # Example generation
    "generate_example_usage",
    # Dataset creation
    "create_lexical_dataset",
]
