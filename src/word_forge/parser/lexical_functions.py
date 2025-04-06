# filepath: /home/lloyd/eidosian_forge/word_forge/src/word_forge/parser/lexical_functions.py
# ============================================================================
#                              IMPORTS
# ============================================================================
import functools
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union, cast

import nltk  # type: ignore
from nltk.corpus import wordnet as wn  # type: ignore
from rdflib import Graph

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
from word_forge.parser.language_model import ModelState

# Download NLTK data quietly
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


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
) -> Iterator[Optional[Any]]:
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
            yield f
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
@functools.lru_cache(maxsize=1024)
def get_synsets(word: str) -> List[Any]:
    """
    Retrieve synsets from WordNet for a given word with efficient caching.

    Args:
        word: Word to look up in WordNet

    Returns:
        List of WordNet synsets for the word
    """
    return wn.synsets(word)  # type: ignore


def get_wordnet_data(word: str) -> List[WordnetEntry]:
    """
    Extract comprehensive linguistic data from WordNet for a given word.

    Args:
        word: Word to retrieve data for

    Returns:
        List of structured entries containing definitions, examples, synonyms, antonyms,
        and part-of-speech information
    """
    results: List[WordnetEntry] = []
    synsets = get_synsets(word)

    for synset in synsets:
        lemmas = synset.lemmas()
        synonyms = [lemma.name().replace("_", " ") for lemma in lemmas]

        # Extract antonyms from lemmas
        antonyms: List[str] = []
        for lemma in lemmas:
            for antonym in lemma.antonyms():
                antonym_name = antonym.name().replace("_", " ")
                if antonym_name not in antonyms:
                    antonyms.append(antonym_name)

        results.append(
            {
                "word": word,
                "definition": synset.definition(),
                "examples": synset.examples(),
                "synonyms": synonyms,
                "antonyms": antonyms,
                "part_of_speech": synset.pos(),
            }
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
    if not isinstance(odict_data, dict):
        return default_entry

    return cast(DictionaryEntry, odict_data.get(word, default_entry))


def get_dbnary_data(word: str, dbnary_path: str) -> List[DbnaryEntry]:
    """
    Extract linguistic data from DBnary RDF for a given word.

    Args:
        word: Word to retrieve data for
        dbnary_path: Path to the DBnary TTL file

    Returns:
        List of entries containing definitions and translations

    Raises:
        LexicalResourceError: If there's an error processing the DBnary data
    """
    if not file_exists(dbnary_path):
        return []

    try:
        graph = Graph()
        graph.parse(dbnary_path, format="ttl")

        sparql_query = f"""
        PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?definition ?translation
        WHERE {{
          ?entry ontolex:canonicalForm/ontolex:writtenRep "{word}"@en .
          OPTIONAL {{ ?entry ontolex:definition/rdfs:label ?definition . }}
          OPTIONAL {{ ?entry ontolex:translation/rdfs:label ?translation . }}
        }}
        """

        results = graph.query(sparql_query)
        output: List[DbnaryEntry] = []

        for row in results:
            # Cast row to Any to bypass type checking for RDFLib query results
            row_any = cast(Any, row)

            definition = str(row_any[0]) if row_any[0] is not None else ""
            translation = str(row_any[1]) if row_any[1] is not None else ""

            if definition or translation:
                output.append({"definition": definition, "translation": translation})

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
    if not isinstance(data, dict):
        return default_entry

    return cast(DictionaryEntry, data.get(word, default_entry))


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
            return data.get("synonyms", [])
        return None

    results: List[str] = []
    for syns in read_jsonl_file(thesaurus_path, process_line):
        results.extend(syns)

    return results


# ============================================================================
#                       EXAMPLE GENERATION FUNCTIONS
# ============================================================================
def generate_example_usage(
    word: str, definition: str, synonyms: List[str], antonyms: List[str], pos: str
) -> str:
    """
    Generate an example sentence for a word using a language model.

    Args:
        word: The target word to use in the example
        definition: The word's definition
        synonyms: List of word synonyms
        antonyms: List of word antonyms
        pos: Part of speech

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
        f"Task: Generate a single concise example sentence using the word '{word}'.\n"
        f"Example Sentence: "
    )

    # Use the improved model generation method
    full_text = ModelState.generate_text(prompt)

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

    Returns:
        Dictionary containing comprehensive lexical data from all sources
    """
    wordnet_data = get_wordnet_data(word)

    dataset: LexicalDataset = {
        "word": word,
        "wordnet_data": wordnet_data,
        "openthesaurus_synonyms": get_openthesaurus_data(word, openthesaurus_path),
        "odict_data": get_odict_data(word, odict_path),
        "dbnary_data": get_dbnary_data(word, dbnary_path),
        "opendict_data": get_opendictdata(word, opendict_path),
        "thesaurus_synonyms": get_thesaurus_data(word, thesaurus_path),
        "example_sentence": "",
    }

    # Generate an example sentence if WordNet data exists
    if wordnet_data:
        first_entry = wordnet_data[0]
        example = generate_example_usage(
            word,
            definition=first_entry.get("definition", ""),
            synonyms=dataset["openthesaurus_synonyms"],
            antonyms=first_entry.get("antonyms", []),
            pos=first_entry.get("part_of_speech", ""),
        )
        dataset["example_sentence"] = example
    else:
        dataset["example_sentence"] = (
            "No example available due to missing WordNet data."
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
