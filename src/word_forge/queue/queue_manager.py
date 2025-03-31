import queue
from typing import Callable, Dict, Generic, Iterator, Optional, Protocol, Set, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class Normalizable(Protocol):
    """Protocol defining objects that can be normalized to a canonical form."""

    def __str__(self) -> str: ...


class QueueError(Exception):
    """Base exception for queue operations."""

    pass


class EmptyQueueError(QueueError):
    """Raised when attempting to dequeue from an empty queue."""

    pass


class QueueManager(Generic[T]):
    """
    Manages a FIFO queue of items to process with duplicate prevention.

    This generic implementation maintains queue order while ensuring each
    item is processed exactly once through efficient tracking.
    """

    def __init__(self, normalize_func: Optional[Callable[[T], T]] = None):
        """
        Initialize an empty queue with tracking for seen items.

        Args:
            normalize_func: Optional function to normalize items before processing.
                Defaults to lowercase string normalization if None.
        """
        self._queue: queue.Queue[T] = queue.Queue()
        self._seen_items: Set[T] = set()
        self._normalize: Callable[[T], T] = normalize_func or self._default_normalize

    def _default_normalize(self, item: T) -> T:
        """
        Default normalization for string items.

        Args:
            item: The item to normalize

        Returns:
            Normalized version of the item (lowercase string with whitespace stripped)
        """
        if isinstance(item, str):
            return item.strip().lower()  # type: ignore
        return item

    def enqueue(self, item: T) -> bool:
        """
        Add an item to the queue if it hasn't already been seen.

        Args:
            item: The item to enqueue

        Returns:
            bool: True if the item was enqueued, False if it was already seen

        Raises:
            ValueError: If the item is empty or None
        """
        self._validate_item(item)
        normalized_item = self._normalize(item)

        if normalized_item not in self._seen_items:
            self._queue.put(normalized_item)
            self._seen_items.add(normalized_item)
            return True
        return False

    def _validate_item(self, item: T) -> None:
        """
        Validate an item before enqueuing.

        Args:
            item: The item to validate

        Raises:
            ValueError: If the item is empty or None
        """
        if item is None:
            raise ValueError("Cannot enqueue None items")
        if isinstance(item, str) and not item.strip():
            raise ValueError("Cannot enqueue empty string items")

    def dequeue(self) -> T:
        """
        Remove and return the next item from the queue.

        Returns:
            The next item from the queue

        Raises:
            EmptyQueueError: If the queue is empty
        """
        if self.is_empty():
            raise EmptyQueueError("Cannot dequeue from an empty queue")
        return self._queue.get()

    def peek(self) -> Optional[T]:
        """
        View the next item without removing it from the queue.

        Returns:
            The next item or None if queue is empty
        """
        if self.is_empty():
            return None

        # Get item, put it back, and return a copy
        item = self._queue.get()
        self._queue.put(item)
        return item

    def size(self) -> int:
        """
        Get the current number of items waiting in the queue.

        Returns:
            Current queue size (excluding already processed items)
        """
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """
        Check if the queue is empty.

        Returns:
            True if no items remain in the queue, False otherwise
        """
        return self._queue.empty()

    def has_seen(self, item: T) -> bool:
        """
        Check if an item has been previously enqueued.

        Args:
            item: The item to check

        Returns:
            True if the normalized item has been seen before, False otherwise
        """
        normalized_item = self._normalize(item)
        return normalized_item in self._seen_items

    def iter_seen(self) -> Iterator[T]:
        """
        Get an iterator over all seen items.

        Returns:
            Iterator of all items that have been enqueued
        """
        return iter(self._seen_items)

    def reset(self) -> None:
        """
        Clear the queue and the set of seen items.

        This method resets the queue to its initial state. Use with caution
        as it eliminates all tracking of previously processed items.
        """
        while not self._queue.empty():
            self._queue.get()
        self._seen_items.clear()

    def batch_enqueue(self, items: Iterator[T]) -> Dict[T, bool]:
        """
        Enqueue multiple items at once and return their success status.

        Args:
            items: Iterator of items to enqueue

        Returns:
            Dictionary mapping items to their enqueue success status
        """
        results: Dict[T, bool] = {}
        for item in items:
            try:
                results[item] = self.enqueue(item)
            except ValueError:
                results[item] = False
        return results


# Backward compatibility aliases
def enqueue_word(self: QueueManager[str], term: str) -> bool:
    """Backward compatibility alias for enqueue."""
    return self.enqueue(term)


def dequeue_word(self: QueueManager[str]) -> Optional[str]:
    """Backward compatibility alias for dequeue."""
    try:
        return self.dequeue()
    except EmptyQueueError:
        return None


def queue_size(self: QueueManager[T]) -> int:
    """Backward compatibility alias for size."""
    return self.size()


# Add backward compatibility methods to QueueManager class
QueueManager.enqueue_word = enqueue_word  # type: ignore
QueueManager.dequeue_word = dequeue_word  # type: ignore
QueueManager.queue_size = queue_size  # type: ignore


def main() -> None:
    """
    Demonstrate the usage of QueueManager with a complete workflow example.
    """
    # Initialize queue manager
    word_queue = QueueManager[str]()

    # Add words to the queue
    print("=== Enqueuing Words ===")
    words = ["Python", "java", "PYTHON", "JavaScript", "TypeScript", "python"]

    # Demonstrate batch enqueue
    results = word_queue.batch_enqueue(iter(words))
    for word, success in results.items():
        print(f"Enqueued '{word}': {success}")

    # Show queue status
    print(f"\nQueue size: {word_queue.size()}")
    print(f"Total unique items seen: {len(list(word_queue.iter_seen()))}")

    # Process queue
    print("\n=== Processing Queue ===")
    while not word_queue.is_empty():
        item = word_queue.dequeue()
        print(f"Processing: {item}")

    # Check for emptiness
    print(f"\nQueue is empty: {word_queue.is_empty()}")

    # Demonstrate has_seen functionality
    print("\n=== Checking Seen Words ===")
    check_words = ["python", "JAVA", "Ruby", "C#"]
    for word in check_words:
        print(f"Has seen '{word}': {word_queue.has_seen(word)}")


# Export public elements
__all__ = [
    "QueueManager",
    "QueueError",
    "EmptyQueueError",
    "Normalizable",
]


if __name__ == "__main__":
    main()
