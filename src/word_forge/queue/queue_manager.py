import queue
import threading
from typing import (
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    cast,
)

T = TypeVar("T")  # Item type
NormalizerFunc = Callable[[T], T]  # Type alias for normalization functions


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
    item is processed exactly once through efficient tracking. The class
    is thread-safe, using reentrant locks to protect shared state access.

    Typical usage:
        queue = QueueManager[str]()
        queue.enqueue("item1")
        item = queue.dequeue()  # Returns "item1"
    """

    def __init__(self, normalize_func: Optional[NormalizerFunc[T]] = None):
        """
        Initialize an empty queue with tracking for seen items.

        Args:
            normalize_func: Optional function to normalize items before processing.
                Defaults to lowercase string normalization if None.
        """
        self._queue: queue.Queue[T] = queue.Queue()
        self._seen_items: Set[T] = set()
        self._normalize: NormalizerFunc[T] = normalize_func or self._default_normalize
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    def _default_normalize(self, item: T) -> T:
        """
        Default normalization for string items.

        Args:
            item: The item to normalize

        Returns:
            Normalized version of the item (lowercase string with whitespace stripped)
        """
        if isinstance(item, str):
            return cast(T, str.strip(str.lower(item)))
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

        with self._lock:
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

        # Thread-safe peek operation to avoid race conditions
        with self._lock:
            try:
                # Get item but don't mark as processed
                item = self._queue.get(block=False)
                # Put it back at the front
                self._queue.put(item)
                return item
            except queue.Empty:
                # Queue became empty between our check and get
                return None

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

    def seen_count(self) -> int:
        """
        Get the total number of unique items that have been enqueued.

        Returns:
            Count of all unique items ever enqueued
        """
        return len(self._seen_items)

    def reset(self) -> None:
        """
        Clear the queue and the set of seen items.

        This method resets the queue to its initial state. Use with caution
        as it eliminates all tracking of previously processed items.
        """
        with self._lock:
            self._queue = queue.Queue()
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

    def get_sample(self, n: int) -> List[T]:
        """
        Get a sample of up to n items currently in the queue without removing them.

        Args:
            n: Maximum number of items to retrieve

        Returns:
            List of up to n items from the queue, or empty list if queue is empty
        """
        if self.is_empty() or n <= 0:
            return []

        with self._lock:
            temp_items: List[T] = []
            sample_size = min(n, self.size())

            # Extract items (up to n)
            for _ in range(sample_size):
                if self.is_empty():
                    break
                item = self._queue.get()
                temp_items.append(item)

            # Put all items back into the queue
            for item in temp_items:
                self._queue.put(item)

            return temp_items[:n]


# Backward compatibility aliases - type signature enhanced but functionality preserved
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
setattr(QueueManager, "enqueue_word", enqueue_word)
setattr(QueueManager, "dequeue_word", dequeue_word)
setattr(QueueManager, "queue_size", queue_size)


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
    print(f"Total unique items seen: {word_queue.seen_count()}")

    # Demonstrate sample functionality
    if word_queue.size() > 0:
        print("\n=== Queue Sample ===")
        sample = word_queue.get_sample(3)
        for idx, item in enumerate(sample, 1):
            print(f"Sample item {idx}: {item}")

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
