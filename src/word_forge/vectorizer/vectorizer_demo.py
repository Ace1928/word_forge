"""
Vector Embedding Demonstration for Word Forge.

This module demonstrates the complete vector embedding workflow from text to search,
showcasing the multilingual capabilities and semantic search features of the system.
It serves as both a functional example and a testing tool for the vector components.

Usage:
    python -m word_forge.vectorizer.vectorizer_demo
"""

# ... (rest of the existing code in vector_demo.py) ...


def main() -> None:
    """
    Main entry point for running the vector demo.

    Creates a demo instance, adds multilingual examples, and
    performs sample searches to showcase capabilities.

    Contains error handling to ensure the demo runs successfully
    even if some operations fail.
    """
    try:
        # Initialize demo
        demo = VectorDemo()

        # Add test data
        demo.add_multilingual_examples()

        # Perform sample searches
        demo.search_similar("algorithm for solving problems")
        demo.search_similar("递归技术", k=5)  # "recursive technique" in Chinese
        demo.search_similar(
            "procédure étape par étape"
        )  # "step by step procedure" in French

        # Demonstrate language filtering
        demo.search_similar("algorithm", filter_language="zh")

    except Exception as e:
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
