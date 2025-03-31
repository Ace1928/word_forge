import argparse
import datetime
import logging
import os
import sys
import tempfile
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, cast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Optional: for colorful/stylish output
try:
    from colorama import Fore, Style
    from colorama import init as colorama_init

    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    # If colorama is not installed, just ignore (no color)
    class _NoColor:
        def __getattr__(self, _):
            return ""

    Fore = Style = _NoColor()
    HAS_COLOR = False

# For optional argument auto-completion in bash/zsh/fish:
# 1) pip install argcomplete
# 2) eval "$(register-python-argcomplete <your_script_name>)"
try:
    import argcomplete

    HAS_ARGCOMPLETION = True
except ImportError:
    HAS_ARGCOMPLETION = False
    argcomplete = None  # Define a null value to prevent unbound issues

# WordForge modules
from word_forge.database.db_manager import DBManager, WordEntryDict
from word_forge.parser.parser_refiner import ParserRefiner
from word_forge.queue.queue_manager import QueueManager
from word_forge.worker.worker_thread import (
    ProcessingResult,
    WordForgeWorker,
    create_progress_bar,
    print_table,
)


class CLIState(Enum):
    """States for the WordForgeCLI application lifecycle."""

    INITIALIZING = auto()
    RUNNING = auto()
    INTERACTIVE = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class CLICommand(Enum):
    """Available commands for the WordForgeCLI."""

    START = "start"
    STOP = "stop"
    ADD_WORD = "add-word"
    SEARCH = "search"
    QUEUE = "queue"
    STATS = "stats"
    INTERACTIVE = "interactive"
    BATCH = "batch"
    HELP = "help"


class WordForgeCLI:
    """
    A fully Eidosian CLI for managing the living lexical database,
    providing subcommands and an optional interactive mode.
    """

    def __init__(
        self,
        db_path: str = "word_forge.sqlite",
        data_dir: str = "data",
        log_level: str = "INFO",
    ):
        """
        Initialize the WordForgeCLI application.

        Args:
            db_path: Path to the SQLite database file
            data_dir: Directory containing lexical resources
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Configure logging
        self.logger = logging.getLogger("WordForgeCLI")
        self.logger.setLevel(getattr(logging, log_level))

        # Initialize components
        self._init_components(db_path, data_dir)

        # State tracking
        self.state = CLIState.INITIALIZING
        self.worker_started = False
        self._last_stats_time = 0
        self._stats_interval = 5  # seconds
        self._exit_event = threading.Event()

    def _init_components(self, db_path: str, data_dir: str) -> None:
        """
        Initialize the core components of WordForge.

        Args:
            db_path: Path to the SQLite database file
            data_dir: Directory containing lexical resources
        """
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Initialize components
        self.db_path = db_path
        self.data_dir = data_dir
        self.db_manager = DBManager(db_path=db_path)
        # Explicitly type the queue manager
        self.queue_manager = cast(QueueManager[str], QueueManager())
        self.parser_refiner = ParserRefiner(
            db_manager=self.db_manager,
            queue_manager=self.queue_manager,
            data_dir=data_dir,
            enable_model=True,
        )

        # Initialize worker with processing result callback
        self.worker = WordForgeWorker(
            self.parser_refiner,
            self.queue_manager,
            sleep_interval=0.5,
            result_callback=self._on_word_processed,
            logger=self.logger,
        )

        # Statistics
        self.processed_words: Set[str] = set()
        self.start_time = 0.0

        self.state = CLIState.RUNNING
        self.logger.debug("WordForgeCLI initialized")

    def _on_word_processed(self, result: ProcessingResult) -> None:
        """
        Handle word processing results from the worker.

        Args:
            result: Processing result data
        """
        term = result["term"]
        success = result["success"]

        if success:
            self.processed_words.add(term)

            # Log processing result at appropriate intervals
            current_time = time.time()
            if current_time - self._last_stats_time >= self._stats_interval:
                self._last_stats_time = current_time
                stats = self.worker.get_statistics()

                proc_count = stats.get("processed_count", 0)
                proc_rate = stats.get("processing_rate_per_minute", 0.0)
                queue_size = stats.get("queue_size", 0)

                if HAS_COLOR:
                    status_line = (
                        f"{Fore.GREEN}Processed {proc_count} words "
                        f"{Fore.CYAN}({proc_rate:.1f}/min) | "
                        f"{Fore.YELLOW}Queue: {queue_size}{Style.RESET_ALL}"
                    )
                else:
                    status_line = (
                        f"Processed {proc_count} words "
                        f"({proc_rate:.1f}/min) | "
                        f"Queue: {queue_size}"
                    )

                self.logger.info(status_line)

    def start_worker(self) -> None:
        """Start the worker thread if not already running."""
        if not self.worker_started:
            self.worker_started = True
            self.start_time = time.time()
            self.logger.info(f"{Fore.GREEN}Starting worker thread...{Style.RESET_ALL}")
            self.worker.start()
        else:
            self.logger.info(
                f"{Fore.YELLOW}Worker thread already running.{Style.RESET_ALL}"
            )

    def stop_worker(self) -> None:
        """Stop the worker thread if running and wait for it to finish."""
        if self.worker_started:
            self.logger.info(f"{Fore.GREEN}Stopping worker thread...{Style.RESET_ALL}")
            self.worker.request_stop()
            self.worker.join(timeout=5.0)
            self.worker_started = False

            # Print final statistics
            if self.start_time > 0:
                runtime = time.time() - self.start_time
                formatted_runtime = str(datetime.timedelta(seconds=int(runtime)))
                stats = self.worker.get_statistics()

                self.logger.info(
                    f"\n{Fore.CYAN}=== Processing Summary ==={Style.RESET_ALL}"
                )
                self.logger.info(f"Runtime: {formatted_runtime}")
                self.logger.info(f"Words processed: {stats.get('processed_count', 0)}")
                self.logger.info(
                    f"Processing rate: {stats.get('processing_rate_per_minute', 0.0):.2f} words/minute"
                )
                self.logger.info(f"Words in queue: {stats.get('queue_size', 0)}")
                self.logger.info(
                    f"Unique words seen: {stats.get('total_unique_words', 0)}"
                )
                self.logger.info(f"Error count: {stats.get('error_count', 0)}")
        else:
            self.logger.info(
                f"{Fore.YELLOW}Worker thread is not running.{Style.RESET_ALL}"
            )

    def add_word(self, term: str) -> bool:
        """
        Enqueue a word for processing.

        Args:
            term: The word to add to the processing queue

        Returns:
            Whether the word was successfully enqueued
        """
        enqueued = self.queue_manager.enqueue(term)
        if enqueued:
            self.logger.info(
                f"{Fore.GREEN}Enqueued '{term}' for processing.{Style.RESET_ALL}"
            )
        else:
            self.logger.info(
                f"{Fore.YELLOW}'{term}' is already queued or processed.{Style.RESET_ALL}"
            )
        return enqueued

    def batch_add_words(self, file_path: str) -> Tuple[int, int]:
        """
        Add multiple words from a file (one word per line).

        Args:
            file_path: Path to file containing words

        Returns:
            Tuple of (enqueued count, total word count)
        """
        try:
            with open(file_path, "r") as f:
                words = [line.strip() for line in f if line.strip()]

            self.logger.info(f"Processing {len(words)} words from {file_path}")

            enqueued_count = 0
            for word in words:
                if self.add_word(word):
                    enqueued_count += 1

            return enqueued_count, len(words)

        except Exception as e:
            self.logger.error(f"Error reading batch file: {str(e)}")
            return 0, 0

    def search_word(self, term: str) -> Optional[WordEntryDict]:
        """
        Retrieve and display a word entry from the database.

        Args:
            term: The word to search for

        Returns:
            The word entry if found, None otherwise
        """
        try:
            entry = self.db_manager.get_word_if_exists(term.lower())
            if not entry:
                self.logger.info(
                    f"{Fore.RED}No entry found for '{term}'.{Style.RESET_ALL}"
                )
                return None

            # Display the entry in a user-friendly manner
            self.logger.info(
                f"\n{Fore.CYAN}=== Word Entry: {term} ==={Style.RESET_ALL}"
            )
            self.logger.info(f"Definition: {entry.get('definition', '')}")
            self.logger.info(f"Part of Speech: {entry.get('part_of_speech', '')}")

            examples = entry.get("usage_examples", [])
            if examples:
                self.logger.info("Usage Examples:")
                for idx, ex in enumerate(examples, 1):
                    self.logger.info(f"  {idx}. {ex}")

            rels = entry.get("relationships", [])
            if rels:
                # Group relationships by type
                rel_by_type: Dict[str, List[str]] = {}
                for r in rels:
                    rel_type = r["relationship_type"]
                    if rel_type not in rel_by_type:
                        rel_by_type[rel_type] = []
                    rel_by_type[rel_type].append(r["related_term"])

                self.logger.info("Relationships:")
                for rel_type, terms in rel_by_type.items():
                    term_list = ", ".join(terms[:5])
                    remaining = len(terms) - 5
                    if remaining > 0:
                        term_list += f" (+{remaining} more)"
                    self.logger.info(f"  - {rel_type}: {term_list}")

            return entry

        except Exception as e:
            self.logger.error(f"Error searching for '{term}': {str(e)}")
            return None

    def list_queue(self) -> int:
        """
        Display the current queue status.

        Returns:
            The number of items in the queue
        """
        size = self.queue_manager.size()
        self.logger.info(f"{Fore.BLUE}Queue has {size} items pending.{Style.RESET_ALL}")
        return size

    def display_statistics(self) -> None:
        """Display detailed statistics about the worker and database."""
        if not self.worker_started:
            self.logger.info(
                f"{Fore.YELLOW}Worker not running. No statistics available.{Style.RESET_ALL}"
            )
            return

        stats = self.worker.get_statistics()

        # Print worker status
        state_str = stats.get("state", "UNKNOWN")
        status_color = {
            "RUNNING": Fore.GREEN,
            "PAUSED": Fore.YELLOW,
            "STOPPING": Fore.RED,
            "STOPPED": Fore.RED,
            "ERROR": Fore.RED,
        }.get(state_str, Fore.WHITE)

        # Format runtime
        runtime_secs = stats.get("runtime_seconds", 0)
        if isinstance(runtime_secs, (int, float)):
            formatted_runtime = str(datetime.timedelta(seconds=int(runtime_secs)))
        else:
            formatted_runtime = "Unknown"

        # Create and print statistics table
        headers = ["Metric", "Value"]
        rows = []

        def format_stat(key: str, default: str, formatter: Optional[str] = None) -> str:
            """Format a statistic with proper type handling"""
            value = stats.get(key, default)
            if formatter and isinstance(value, (int, float)):
                if formatter == "int":
                    return str(int(value))
                elif formatter == "float1":
                    return f"{value:.1f}"
                elif formatter == "float2":
                    return f"{value:.2f}"
                elif formatter == "float3":
                    return f"{value:.3f}"
                elif formatter == "ms":
                    return f"{value * 1000:.2f} ms"
            return str(value)

        rows = [
            ["Status", f"{status_color}{state_str}{Style.RESET_ALL}"],
            ["Runtime", formatted_runtime],
            ["Words Processed", format_stat("processed_count", "0", "int")],
            [
                "Processing Rate",
                format_stat("processing_rate_per_minute", "0.0", "float2")
                + " words/min",
            ],
            ["Avg Processing Time", format_stat("avg_processing_time", "0.0", "ms")],
            ["Queue Size", format_stat("queue_size", "0", "int")],
            ["Total Unique Words", format_stat("total_unique_words", "0", "int")],
            ["Error Count", format_stat("error_count", "0", "int")],
            ["Last Processed", format_stat("last_processed", "None")],
        ]

        self.logger.info(f"\n{Fore.CYAN}=== WordForge Statistics ==={Style.RESET_ALL}")
        print_table(headers, rows)

        # Show progress bar if processing
        if state_str == "RUNNING":
            processed = int(stats.get("processed_count", 0))
            queue_size = int(stats.get("queue_size", 0))
            if processed > 0 and queue_size > 0:
                total = processed + queue_size
                self.logger.info(f"\nProgress: {create_progress_bar(processed, total)}")

        # Show recent activities
        recent_events = stats.get("recent_events", [])
        if isinstance(recent_events, list) and recent_events:
            self.logger.info(f"\n{Fore.CYAN}Recent Activity:{Style.RESET_ALL}")
            for event in recent_events[-3:]:  # Show last 3 events
                if not isinstance(event, dict):
                    continue

                event_time_raw = event.get("timestamp")
                event_type = event.get("event_type")
                term = event.get("term", "unknown")
                error = event.get("error", "unknown error")

                # Safe timestamp conversion
                if isinstance(event_time_raw, (int, float)):
                    event_time = datetime.datetime.fromtimestamp(
                        event_time_raw
                    ).strftime("%H:%M:%S")
                else:
                    event_time = "??:??:??"

                if event_type == "word_processed":
                    self.logger.info(f"  {event_time} - Processed: {term}")
                elif event_type == "error":
                    self.logger.info(f"  {event_time} - Error: {error}")
                elif event_type == "state_change":
                    details = event.get("details", {})
                    if isinstance(details, dict):
                        old_state = details.get("old_state", "?")
                        new_state = details.get("new_state", "?")
                        self.logger.info(
                            f"  {event_time} - State changed: {old_state} → {new_state}"
                        )

    def interactive_shell(self) -> None:
        """
        Launch an interactive REPL-like shell for WordForge.

        Commands:
          start      - start the worker
          stop       - stop the worker
          add <word> - enqueue a word
          search <word> - lookup word in DB
          queue      - show queue size
          stats      - show detailed statistics
          exit       - stop worker & quit
        """
        self.state = CLIState.INTERACTIVE
        print(
            f"{Fore.GREEN}Entering WordForge Interactive Mode. Type 'help' for commands.{Style.RESET_ALL}"
        )
        self.start_worker()  # Start worker by default in interactive mode

        while True:
            try:
                cmd = input(f"\n{Fore.MAGENTA}WordForge> {Style.RESET_ALL}").strip()
            except (EOFError, KeyboardInterrupt):
                cmd = "exit"

            if not cmd:
                continue

            parts = cmd.split(maxsplit=1)
            main_cmd = parts[0].lower()

            if main_cmd == "help":
                self._display_help()
            elif main_cmd == "start":
                self.start_worker()
            elif main_cmd == "stop":
                self.stop_worker()
            elif main_cmd == "add":
                if len(parts) < 2:
                    print("Usage: add <word>")
                else:
                    word = parts[1]
                    self.add_word(word)
            elif main_cmd == "batch":
                if len(parts) < 2:
                    print("Usage: batch <file_path>")
                else:
                    file_path = parts[1]
                    enqueued, total = self.batch_add_words(file_path)
                    print(f"Enqueued {enqueued} of {total} words from {file_path}")
            elif main_cmd == "search":
                if len(parts) < 2:
                    print("Usage: search <word>")
                else:
                    word = parts[1]
                    self.search_word(word)
            elif main_cmd == "queue":
                self.list_queue()
            elif main_cmd == "stats":
                self.display_statistics()
            elif main_cmd == "exit":
                self.stop_worker()
                print(
                    f"{Fore.GREEN}Exiting WordForge interactive mode.{Style.RESET_ALL}"
                )
                break
            else:
                print(f"{Fore.RED}Unknown command: {main_cmd}{Style.RESET_ALL}")
                print("Type 'help' for a list of commands.")

    def _display_help(self) -> None:
        """Display available commands for the interactive shell."""
        print(
            f"\n{Fore.CYAN}=== WordForge Commands ==={Style.RESET_ALL}\n"
            f"  {Fore.GREEN}start{Style.RESET_ALL}            - Start the worker\n"
            f"  {Fore.GREEN}stop{Style.RESET_ALL}             - Stop the worker\n"
            f"  {Fore.GREEN}add <word>{Style.RESET_ALL}       - Enqueue a word for processing\n"
            f"  {Fore.GREEN}batch <file>{Style.RESET_ALL}     - Enqueue words from a file (one per line)\n"
            f"  {Fore.GREEN}search <word>{Style.RESET_ALL}    - Look up a word in the DB\n"
            f"  {Fore.GREEN}queue{Style.RESET_ALL}            - Show queue size\n"
            f"  {Fore.GREEN}stats{Style.RESET_ALL}            - Show detailed statistics\n"
            f"  {Fore.GREEN}exit{Style.RESET_ALL}             - Stop worker & exit\n"
            f"  {Fore.GREEN}help{Style.RESET_ALL}             - Show this help message\n"
        )

    def shutdown(self) -> None:
        """
        Stop the worker if running, then exit the application.
        """
        self.state = CLIState.STOPPING
        self.stop_worker()
        self.state = CLIState.STOPPED
        print(f"{Fore.CYAN}--- Shutting down WordForge CLI ---{Style.RESET_ALL}")
        sys.exit(0)

    def run_demo(self, minutes: float = 1.0) -> None:
        """
        Run a demonstration of WordForge for a specified duration.

        Args:
            minutes: Duration to run the demo in minutes
        """
        run_seconds = minutes * 60

        # Create some demo files
        temp_dir = tempfile.mkdtemp()
        demo_data_dir = os.path.join(temp_dir, "data")
        os.makedirs(demo_data_dir, exist_ok=True)

        # Create sample dictionary files
        self._create_demo_files(Path(demo_data_dir))

        # Reinitialize with demo data
        self._init_components(
            os.path.join(temp_dir, "word_forge_demo.sqlite"), demo_data_dir
        )

        # Seed the queue with initial words
        seed_words = [
            "algorithm",
            "language",
            "computer",
            "data",
            "program",
            "network",
            "code",
            "function",
            "logic",
            "system",
        ]

        self.logger.info(f"Seeding queue with {len(seed_words)} initial words...")
        for word in seed_words:
            self.add_word(word)

        # Start worker
        self.logger.info(f"Starting worker for {minutes} minute(s)...")
        self.start_worker()

        # Monitor progress during runtime with periodic status updates
        update_interval = 5  # Show stats every 5 seconds
        start_time = time.time()
        end_time = start_time + run_seconds

        try:
            while time.time() < end_time and self.worker_started:
                # Sleep for the update interval or until end time
                remaining = min(update_interval, end_time - time.time())
                if remaining <= 0:
                    break
                time.sleep(remaining)

                # Show progress
                if hasattr(self.worker, "formatted_statistics"):
                    self.logger.info(self.worker.formatted_statistics())

                elapsed = time.time() - start_time
                # Convert to integers to satisfy create_progress_bar's type requirements
                self.logger.info(
                    f"Progress: {create_progress_bar(int(elapsed), int(run_seconds))}"
                )

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user. Stopping gracefully...")

        finally:
            # Display final stats and summary
            self.stop_worker()
            self.logger.info(f"Demo files at: {temp_dir}")
            self.logger.info("Demo complete!")

    def _create_demo_files(self, data_dir: Path) -> None:
        """
        Create sample dictionary files for demonstration.

        Args:
            data_dir: Directory to create files in
        """
        with open(data_dir / "openthesaurus.jsonl", "w") as f:
            f.write('{"words": ["algorithm"], "synonyms": ["procedure", "method"]}\n')
            f.write(
                '{"words": ["language"], "synonyms": ["tongue", "speech", "dialect"]}\n'
            )
            f.write(
                '{"words": ["computer"], "synonyms": ["machine", "processor", "device"]}\n'
            )
            f.write('{"words": ["network"], "synonyms": ["web", "grid", "system"]}\n')
            f.write(
                '{"words": ["data"], "synonyms": ["information", "facts", "figures"]}\n'
            )

        with open(data_dir / "odict.json", "w") as f:
            f.write(
                """{
                "algorithm": {"definition": "A step-by-step procedure", "examples": ["Sorting algorithms are fundamental."]},
                "language": {"definition": "A system of communication", "examples": ["English is a global language."]},
                "computer": {"definition": "An electronic device for processing data", "examples": ["The computer crashed."]},
                "data": {"definition": "Facts and statistics collected together", "examples": ["The data shows an increasing trend."]},
                "network": {"definition": "A group of interconnected systems", "examples": ["The computer network spans multiple buildings."]}
            }"""
            )

        with open(data_dir / "opendict.json", "w") as f:
            f.write("{}")

        with open(data_dir / "thesaurus.jsonl", "w") as f:
            f.write('{"word": "algorithm", "synonyms": ["process", "routine"]}\n')
            f.write(
                '{"word": "language", "synonyms": ["communication", "expression"]}\n'
            )
            f.write('{"word": "program", "synonyms": ["application", "software"]}\n')
            f.write('{"word": "function", "synonyms": ["procedure", "routine"]}\n')


def main() -> None:
    """
    The main entry point for the WordForge CLI. Provides subcommands
    for an all-in-one lexical database solution.
    """
    parser = argparse.ArgumentParser(
        description="WordForge: A Living, Self-Refining Lexical Database."
    )
    parser.add_argument(
        "--db-path",
        default="word_forge.sqlite",
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing lexical resource files (JSON, TTL, etc.).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    subparsers = parser.add_subparsers(
        title="Commands", description="Available operations", dest="command"
    )

    # Subcommand: start (starts the background worker)
    start_parser = subparsers.add_parser(
        "start", help="Start the worker in background."
    )
    start_parser.add_argument(
        "--minutes",
        type=float,
        default=0,
        help="Run for specified minutes then stop (0 for indefinite).",
    )

    # Subcommand: stop (stops the background worker)
    subparsers.add_parser("stop", help="Stop the background worker.")

    # Subcommand: add-word
    add_parser = subparsers.add_parser(
        "add-word", help="Enqueue a word for processing."
    )
    add_parser.add_argument("word", help="The word to enqueue.")

    # Subcommand: batch
    batch_parser = subparsers.add_parser(
        "batch", help="Add multiple words from a file."
    )
    batch_parser.add_argument("file", help="Path to file with words (one per line).")

    # Subcommand: search
    search_parser = subparsers.add_parser("search", help="Search for a word in the DB.")
    search_parser.add_argument("word", help="The word to look up.")

    # Subcommand: queue
    subparsers.add_parser("queue", help="Show the current queue size.")

    # Subcommand: stats
    subparsers.add_parser("stats", help="Show detailed statistics.")

    # Subcommand: interactive
    subparsers.add_parser("interactive", help="Enter interactive mode.")

    # Subcommand: demo
    demo_parser = subparsers.add_parser(
        "demo", help="Run a demonstration with sample data."
    )
    demo_parser.add_argument(
        "--minutes", type=float, default=1.0, help="Duration of the demo in minutes."
    )

    if HAS_ARGCOMPLETION and argcomplete is not None:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()
    cli = WordForgeCLI(
        db_path=args.db_path, data_dir=args.data_dir, log_level=args.log_level
    )

    # If no subcommand provided, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to subcommands
    try:
        if args.command == "start":
            # Add seed words to ensure there's something to process
            seed_words = [
                "algorithm",
                "knowledge",
                "intelligence",
                "system",
                "recursive",
            ]
            for word in seed_words:
                cli.add_word(word)

            cli.start_worker()

            # If time-limited run was requested
            if args.minutes > 0:
                end_time = time.time() + (args.minutes * 60)
                try:
                    while time.time() < end_time and not cli._exit_event.is_set():
                        # Show statistics every 10 seconds
                        cli.display_statistics()
                        time.sleep(10)
                    cli.stop_worker()
                except KeyboardInterrupt:
                    print("\nInterrupted by user.")
                    cli.stop_worker()
            else:
                # Indefinite run with stats display until interrupted
                print("Worker started. Press Ctrl+C to stop.")
                try:
                    while not cli._exit_event.is_set():
                        cli.display_statistics()
                        time.sleep(10)
                except KeyboardInterrupt:
                    print("\nInterrupted by user.")
                    cli.stop_worker()

        elif args.command == "stop":
            cli.stop_worker()

        elif args.command == "add-word":
            cli.start_worker()  # Ensure worker is running
            cli.add_word(args.word)
            if cli.worker_started:
                cli.stop_worker()

        elif args.command == "batch":
            cli.start_worker()  # Ensure worker is running
            enqueued, total = cli.batch_add_words(args.file)
            print(f"Enqueued {enqueued} of {total} words from {args.file}")
            if cli.worker_started:
                cli.stop_worker()

        elif args.command == "search":
            cli.search_word(args.word)

        elif args.command == "queue":
            cli.list_queue()

        elif args.command == "stats":
            if not cli.worker_started:
                cli.start_worker()
            cli.display_statistics()
            if cli.worker_started:
                cli.stop_worker()

        elif args.command == "interactive":
            cli.interactive_shell()

        elif args.command == "demo":
            cli.run_demo(args.minutes)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cli.shutdown()


if __name__ == "__main__":
    main()
