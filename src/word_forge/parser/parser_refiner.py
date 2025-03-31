# -- BEGIN Embedded Lexical Lookup Logic -- #
import functools
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)

import nltk
import torch
from nltk.corpus import wordnet as wn
from rdflib import Graph
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from word_forge.database.db_manager import DBManager
from word_forge.queue.queue_manager import QueueManager

# Download NLTK data quietly
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class LexicalResourceError(Exception):
    """Exception raised when a lexical resource cannot be accessed or processed."""

    pass


class ResourceNotFoundError(LexicalResourceError):
    """Exception raised when a lexical resource cannot be found."""

    pass


class ResourceParsingError(LexicalResourceError):
    """Exception raised when a lexical resource cannot be parsed."""

    pass


class ModelError(Exception):
    """Exception raised when there's an issue with the language model."""

    pass


T = TypeVar("T")
JsonData = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


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
    file_path: Union[str, Path], process_func: Callable[[Dict[str, Any]], Any]
) -> List[Any]:
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
    results = []
    with safe_open(file_path) as fh:
        if fh is None:
            return results

        line_num = 0
        try:
            for line_num, line in enumerate(fh, 1):
                try:
                    data = json.loads(line)
                    processed = process_func(data)
                    if processed is not None:
                        (
                            results.extend(processed)
                            if isinstance(processed, list)
                            else results.append(processed)
                        )
                except json.JSONDecodeError:
                    pass  # Skip invalid JSON lines silently
        except Exception as e:
            raise LexicalResourceError(
                f"Error processing JSONL file {file_path} (line {line_num}): {str(e)}"
            )

    return results


class WordnetEntry(TypedDict):
    """Type definition for a WordNet entry."""

    word: str
    definition: str
    examples: List[str]
    synonyms: List[str]
    antonyms: List[str]
    part_of_speech: str


@functools.lru_cache(maxsize=1024)
def get_synsets(word: str) -> List[Any]:
    """
    Retrieve synsets from WordNet for a given word with caching.

    Args:
        word: Word to look up in WordNet

    Returns:
        List of WordNet synsets
    """
    return wn.synsets(word)


def get_wordnet_data(word: str) -> List[WordnetEntry]:
    """
    Extract comprehensive linguistic data from WordNet for a given word.

    Args:
        word: Word to retrieve data for

    Returns:
        List of dictionaries containing definitions, examples, synonyms, antonyms, etc.
    """
    results: List[WordnetEntry] = []
    synsets = get_synsets(word)

    for synset in synsets:
        lemmas = synset.lemmas()
        synonyms = [lemma.name() for lemma in lemmas]

        # Extract antonyms from lemmas
        antonyms: List[str] = []
        for lemma in lemmas:
            ants = lemma.antonyms()
            if ants:
                antonyms.append(ants[0].name())

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


def get_openthesaurus_data(word: str, openthesaurus_path: str) -> List[str]:
    """
    Extract synonyms from OpenThesaurus for a given word.

    Args:
        word: Word to retrieve synonyms for
        openthesaurus_path: Path to the OpenThesaurus JSONL file

    Returns:
        List of unique synonyms
    """

    def process_line(d: Dict[str, Any]) -> Optional[List[str]]:
        if word in d.get("words", []):
            return d.get("synonyms", [])
        return None

    synonyms: List[str] = []
    for syns in read_jsonl_file(openthesaurus_path, process_line):
        synonyms.extend(syns)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(synonyms))


class DictionaryEntry(TypedDict):
    """Type definition for a dictionary entry."""

    definition: str
    examples: List[str]


def get_odict_data(word: str, odict_path: str) -> DictionaryEntry:
    """
    Retrieve dictionary data from ODict for a given word.

    Args:
        word: Word to retrieve data for
        odict_path: Path to the ODict JSON file

    Returns:
        Dictionary containing definition and examples
    """
    default_res: DictionaryEntry = {
        "definition": "Not Found",
        "examples": [],
    }
    odict_data = read_json_file(odict_path, {})
    return odict_data.get(word, default_res)


class DbnaryEntry(TypedDict):
    """Type definition for a DBnary entry."""

    definition: str
    translation: str


def get_dbnary_data(word: str, dbnary_path: str) -> List[DbnaryEntry]:
    """
    Extract linguistic data from DBnary RDF for a given word.

    Args:
        word: Word to retrieve data for
        dbnary_path: Path to the DBnary TTL file

    Returns:
        List of dictionaries containing definitions and translations

    Raises:
        LexicalResourceError: If there's an error processing the DBnary data
    """
    if not file_exists(dbnary_path):
        return []

    try:
        graph = Graph()
        graph.parse(dbnary_path, format="ttl")

        query = f"""
        PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?definition ?translation
        WHERE {{
          ?entry ontolex:canonicalForm/ontolex:writtenRep "{word}"@en .
          OPTIONAL {{ ?entry ontolex:definition/rdfs:label ?definition . }}
          OPTIONAL {{ ?entry ontolex:translation/rdfs:label ?translation . }}
        }}
        """

        results = graph.query(query)
        output: List[DbnaryEntry] = []

        for row in results:
            definition_val = str(row[0]) if row[0] else ""
            translation_val = str(row[1]) if row[1] else ""

            output.append(
                {
                    "definition": definition_val,
                    "translation": translation_val,
                }
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
    default_res: DictionaryEntry = {
        "definition": "Not Found",
        "examples": [],
    }
    data = read_json_file(opendict_path, {})
    return data.get(word, default_res)


def get_thesaurus_data(word: str, thesaurus_path: str) -> List[str]:
    """
    Extract synonyms from Thesaurus for a given word.

    Args:
        word: Word to retrieve synonyms for
        thesaurus_path: Path to the Thesaurus JSONL file

    Returns:
        List of synonyms
    """

    def process_line(d: Dict[str, Any]) -> Optional[List[str]]:
        if word == d.get("word"):
            return d.get("synonyms", [])
        return None

    results: List[str] = []
    for syns in read_jsonl_file(thesaurus_path, process_line):
        results.extend(syns)

    return results


class _ModelState:
    """
    Manages transformer model state safely with lazy initialization.

    This singleton class ensures the model is only loaded when needed
    and properly configured for efficient inference.
    """

    _initialized = False
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
    model: Optional[PreTrainedModel] = None
    _model_name: str = "qwen/qwen2.5-0.5b-instruct"
    _device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _inference_failures: int = 0
    _max_failures: int = 5
    _failure_threshold_reached: bool = False

    @classmethod
    def set_model(cls, model_name: str) -> None:
        """
        Set the model to be used for text generation.

        Args:
            model_name: Name of the HuggingFace model to use
        """
        cls._model_name = model_name
        cls._initialized = False  # Force reinitialization

    @classmethod
    def initialize(cls) -> bool:
        """
        Initialize the model and tokenizer if not already done.

        Returns:
            Boolean indicating whether initialization was successful
        """
        if cls._initialized:
            return True

        if cls._failure_threshold_reached:
            return False

        try:
            # Configure torch for optimal inference performance
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

            # Load tokenizer and model with explicit typing
            cls.tokenizer = AutoTokenizer.from_pretrained(cls._model_name)

            # Ensure padding token is set
            if cls.tokenizer.pad_token is None:
                cls.tokenizer.pad_token = cls.tokenizer.eos_token

            # Load model with appropriate device placement
            cls.model = AutoModelForCausalLM.from_pretrained(
                cls._model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            )

            cls._initialized = True
            return True

        except Exception as e:
            cls._inference_failures += 1
            if cls._inference_failures >= cls._max_failures:
                cls._failure_threshold_reached = True
                print(
                    f"Warning: Maximum model initialization failures reached. Disabling model: {str(e)}"
                )
            else:
                print(
                    f"Warning: Error initializing model (attempt {cls._inference_failures}): {str(e)}"
                )
            return False

    @classmethod
    def generate_text(
        cls,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        num_beams: int = 3,
    ) -> Optional[str]:
        """
        Generate text using the loaded model with error handling.

        Args:
            prompt: Input text to generate from
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            num_beams: Number of beams for beam search

        Returns:
            Generated text or None if generation failed
        """
        if not cls.initialize() or cls.tokenizer is None or cls.model is None:
            return None

        try:
            # Create inputs with proper attention mask
            inputs = cls.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(cls._device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(cls._device)

            # Generate with robust error handling for different return types
            with torch.no_grad():
                output = cls.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    temperature=temperature,
                    early_stopping=True,
                    pad_token_id=cls.tokenizer.pad_token_id,
                    do_sample=temperature > 0.0,
                )

            # Handle different output formats from different model versions
            # Some models return a tensor directly, others return an object with sequences attribute
            if hasattr(output, "sequences"):
                generated_ids = output.sequences[0]
            else:
                # Direct tensor output
                generated_ids = output[0] if len(output.shape) > 1 else output

            # Decode the generated text
            generated_text = cls.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            return generated_text

        except Exception as e:
            cls._inference_failures += 1
            print(f"Warning: Error during text generation: {str(e)}")
            return None


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
    full_text = _ModelState.generate_text(prompt)

    if not full_text:
        return f"Could not generate example for '{word}'."

    # Parse out just the generated example
    if "Example Sentence:" in full_text:
        parts = full_text.split("Example Sentence:")
        if len(parts) > 1:
            example = parts[1].strip()
            # Capture up to first period for a complete sentence
            if "." in example:
                return example.split(".")[0] + "."

    # If we got text but couldn't parse it properly, return it as-is
    if full_text and not full_text.startswith("Could not"):
        # Try to find the first complete sentence
        sentences = re.split(r"[.!?]", full_text)
        if sentences and len(sentences[0]) > 5:  # Minimum length for a valid sentence
            return sentences[0].strip() + "."
        return full_text.strip()

    return f"Could not extract valid example for '{word}'."


class LexicalDataset(TypedDict):
    """Type definition for the comprehensive lexical dataset."""

    word: str
    wordnet_data: List[WordnetEntry]
    openthesaurus_synonyms: List[str]
    odict_data: DictionaryEntry
    dbnary_data: List[DbnaryEntry]
    opendict_data: DictionaryEntry
    thesaurus_synonyms: List[str]
    example_sentence: str


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
        "example_sentence": "",  # Will be populated below
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


# -- END Embedded Lexical Lookup Logic -- #


class ParserRefiner:
    """
    Uses advanced lexical lookups to:
    1. Retrieve definitions, synonyms, antonyms, usage examples, etc.
    2. Store them in the DB.
    3. Enqueue new words discovered from that data for further processing.
    """

    def __init__(
        self,
        db_manager: DBManager,
        queue_manager: QueueManager,
        data_dir: Optional[str] = "data",
        enable_model: bool = True,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the ParserRefiner with database and queue managers.

        Args:
            db_manager: DBManager instance for database operations
            queue_manager: QueueManager instance for enqueuing new terms
            data_dir: Path to the folder containing lexical resources
            enable_model: Whether to enable language model for example generation
            model_name: Custom model name to use (if None, uses default)
        """
        self.db_manager = db_manager
        self.queue_manager = queue_manager
        self.data_dir = data_dir or "data"

        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Configure model if needed
        if enable_model and model_name:
            _ModelState.set_model(model_name)

        # Initialize resource paths
        self.resource_paths = {
            "openthesaurus": f"{self.data_dir}/openthesaurus.jsonl",
            "odict": f"{self.data_dir}/odict.json",
            "dbnary": f"{self.data_dir}/dbnary.ttl",
            "opendict": f"{self.data_dir}/opendict.json",
            "thesaurus": f"{self.data_dir}/thesaurus.jsonl",
        }

        # Tracking metrics
        self.processed_count = 0
        self.successful_count = 0
        self.error_count = 0

    def process_word(self, term: str) -> bool:
        """
        Process a word using the integrated lexical resources.

        Data is stored in the DB and new terms are enqueued for further processing.

        Args:
            term: The word to process

        Returns:
            Boolean indicating whether processing was successful
        """
        term_lower = term.strip().lower()
        if not term_lower:
            return False

        try:
            self.processed_count += 1

            # Retrieve comprehensive lexical data
            dataset = create_lexical_dataset(
                term_lower,
                openthesaurus_path=self.resource_paths["openthesaurus"],
                odict_path=self.resource_paths["odict"],
                dbnary_path=self.resource_paths["dbnary"],
                opendict_path=self.resource_paths["opendict"],
                thesaurus_path=self.resource_paths["thesaurus"],
            )

            # Consolidate definitions
            combined_definitions = self._extract_all_definitions(dataset)
            full_definition = (
                " | ".join(combined_definitions) if combined_definitions else ""
            )

            # Determine part_of_speech
            part_of_speech = self._extract_part_of_speech(dataset)

            # Gather usage examples
            usage_examples = self._extract_usage_examples(dataset)

            # Insert or update in DB
            self.db_manager.insert_or_update_word(
                term=term_lower,
                definition=full_definition,
                part_of_speech=part_of_speech,
                usage_examples=usage_examples,
            )

            # Process relationships and discovered terms
            self._process_relationships(term_lower, dataset)
            self._discover_new_terms(term_lower, full_definition, usage_examples)

            self.successful_count += 1
            return True

        except Exception as e:
            self.error_count += 1
            print(f"Error processing word '{term_lower}': {str(e)}")
            return False

    def _extract_all_definitions(self, dataset: LexicalDataset) -> List[str]:
        """Extract and deduplicate definitions from all sources."""
        combined_definitions = []

        # WordNet definitions
        for wn_data in dataset["wordnet_data"]:
            defn = wn_data.get("definition", "")
            if defn and defn not in combined_definitions:
                combined_definitions.append(defn)

        # ODict / OpenDictData
        odict_def = dataset["odict_data"].get("definition", "")
        if (
            odict_def
            and odict_def != "Not Found"
            and odict_def not in combined_definitions
        ):
            combined_definitions.append(odict_def)

        open_dict_def = dataset["opendict_data"].get("definition", "")
        if (
            open_dict_def
            and open_dict_def != "Not Found"
            and open_dict_def not in combined_definitions
        ):
            combined_definitions.append(open_dict_def)

        # Dbnary definitions
        for item in dataset["dbnary_data"]:
            defn = item.get("definition", "")
            if defn and defn not in combined_definitions:
                combined_definitions.append(defn)

        return combined_definitions

    def _extract_part_of_speech(self, dataset: LexicalDataset) -> str:
        """Extract part of speech from WordNet data if available."""
        if dataset["wordnet_data"]:
            return dataset["wordnet_data"][0].get("part_of_speech", "")
        return ""

    def _extract_usage_examples(self, dataset: LexicalDataset) -> List[str]:
        """Extract usage examples from all sources."""
        usage_examples: List[str] = []

        # WordNet examples
        for wn_data in dataset["wordnet_data"]:
            for ex in wn_data.get("examples", []):
                if ex and ex not in usage_examples:
                    usage_examples.append(ex)

        # Add auto-generated example sentence
        auto_ex = dataset["example_sentence"]
        if (
            auto_ex
            and auto_ex not in usage_examples
            and "No example available" not in auto_ex
        ):
            usage_examples.append(auto_ex)

        return usage_examples

    def _process_relationships(self, term: str, dataset: LexicalDataset) -> None:
        """Process and store word relationships."""
        # Process WordNet relationships
        for wn_data in dataset["wordnet_data"]:
            # Synonyms
            for syn in wn_data.get("synonyms", []):
                syn_lower = syn.lower()
                if syn_lower != term:  # Skip self-references
                    self.db_manager.insert_relationship(term, syn_lower, "synonym")
                    self.queue_manager.enqueue_word(syn_lower)

            # Antonyms
            for ant in wn_data.get("antonyms", []):
                ant_lower = ant.lower()
                self.db_manager.insert_relationship(term, ant_lower, "antonym")
                self.queue_manager.enqueue_word(ant_lower)

        # OpenThesaurus synonyms
        for s in dataset["openthesaurus_synonyms"]:
            s_lower = s.lower()
            if s_lower != term:
                self.db_manager.insert_relationship(term, s_lower, "synonym")
                self.queue_manager.enqueue_word(s_lower)

        # Thesaurus synonyms
        for s in dataset["thesaurus_synonyms"]:
            s_lower = s.lower()
            if s_lower != term:
                self.db_manager.insert_relationship(term, s_lower, "synonym")
                self.queue_manager.enqueue_word(s_lower)

        # Translations from DBnary
        for item in dataset["dbnary_data"]:
            translation = item.get("translation", "")
            if translation:
                trans_lower = translation.lower()
                self.db_manager.insert_relationship(term, trans_lower, "translation")
                self.queue_manager.enqueue_word(trans_lower)

    def _discover_new_terms(
        self, term: str, definition: str, examples: List[str]
    ) -> None:
        """Discover and enqueue new terms from definitions and examples."""
        # Combine all text for parsing
        text_to_parse = definition + " " + " ".join(examples)

        # Use regex to extract words (could be enhanced with NLP)
        discovered_terms = {
            word.lower() for word in re.findall(r"\b[a-zA-Z]{3,}\b", text_to_parse)
        }

        # Remove the original term and short words
        discovered_terms.discard(term)

        # Enqueue discovered terms
        for new_term in discovered_terms:
            self.queue_manager.enqueue_word(new_term)

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            "processed": self.processed_count,
            "successful": self.successful_count,
            "errors": self.error_count,
            "queue_size": self.queue_manager.size(),
            "unique_words": len(list(self.queue_manager.iter_seen())),
        }


def main() -> None:
    """
    Demonstrate the functionality of the parser_refiner module.

    This function shows a complete workflow of:
    1. Setting up the database and queue managers
    2. Processing a sample word
    3. Displaying the extracted lexical data
    4. Showing how words discovered during processing are enqueued
    """
    import tempfile
    from pathlib import Path

    # Set up a temporary directory for our demo
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "word_forge_demo.sqlite"

    # Initialize managers
    from word_forge.database.db_manager import DBManager
    from word_forge.queue.queue_manager import QueueManager

    db_manager = DBManager(db_path)
    queue_manager = QueueManager[str]()

    # Create data directory with minimal sample files
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Create minimal sample files
    with open(data_dir / "openthesaurus.jsonl", "w") as f:
        f.write('{"words": ["algorithm"], "synonyms": ["procedure", "method"]}\n')

    with open(data_dir / "odict.json", "w") as f:
        f.write(
            '{"algorithm": {"definition": "A step-by-step procedure", "examples": ["Sorting algorithms are fundamental."]}}'
        )

    with open(data_dir / "opendict.json", "w") as f:
        f.write("{}")

    with open(data_dir / "thesaurus.jsonl", "w") as f:
        f.write('{"word": "algorithm", "synonyms": ["process", "routine"]}\n')

    print("=== Word Forge Parser Refiner Demo ===")
    print(f"Database: {db_path}")
    print(f"Data directory: {data_dir}")

    # Create the refiner and process a word
    refiner = ParserRefiner(
        db_manager,
        queue_manager,
        str(data_dir),
        enable_model=False,  # Disable model for faster demo
    )

    print("\nProcessing word: 'algorithm'...")
    refiner.process_word("algorithm")

    # Display the result from the database
    word_entry = db_manager.get_word_if_exists("algorithm")
    if word_entry:
        print("\n=== Processed Word Entry ===")
        print(f"Term: {word_entry['term']}")
        print(f"Definition: {word_entry['definition']}")
        print(f"Part of Speech: {word_entry['part_of_speech']}")
        print("Usage Examples:")
        for example in word_entry["usage_examples"]:
            print(f"  - {example}")

        print("\nRelationships:")
        for rel in word_entry["relationships"]:
            print(f"  - {rel['relationship_type']}: {rel['related_term']}")

    # Show discovered words that were enqueued
    print("\n=== Discovered Words (Enqueued) ===")
    enqueued_words = list(queue_manager.iter_seen())
    print(f"Total words enqueued: {len(enqueued_words)}")

    # Print a sample of the enqueued words
    sample_size = min(10, len(enqueued_words))
    print(f"Sample: {', '.join(enqueued_words[:sample_size])}")

    # Demonstrate processing the next word in the queue
    if not queue_manager.is_empty():
        next_word = queue_manager.dequeue()
        print(f"\nProcessing next word from queue: '{next_word}'...")
        refiner.process_word(next_word)

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
