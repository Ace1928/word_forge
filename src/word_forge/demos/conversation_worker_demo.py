"""
Demonstration of ConversationWorker functionality.
"""

import logging
import random
import sys
import time
from typing import Any, Dict, Union

from word_forge.conversation.conversation_manager import ConversationManager
from word_forge.conversation.conversation_worker import (
    ConversationTask,
    ConversationWorker,
)
from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.parser.parser_refiner import ParserRefiner
from word_forge.queue.queue_manager import QueueManager


def main() -> None:
    """
    Demonstrate conversation worker functionality.

    Creates a worker instance and simulates conversation processing
    with metrics tracking and lifecycle management.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("conversation_worker_demo")

    # Initialize components
    worker = None

    try:
        # Create database manager
        db_path = "word_forge_demo.sqlite"
        db_manager = DBManager(db_path=db_path)
        db_manager.create_tables()
        logger.info(f"Using database at {db_path}")

        # Create emotion manager
        emotion_manager = EmotionManager(db_manager)

        # Create conversation manager
        conversation_manager = ConversationManager(db_manager, emotion_manager)

        # Create queue manager with a compatible type for ParserRefiner
        queue_manager_for_parser = QueueManager[str]()

        # Create queue manager for conversation tasks
        queue_manager = QueueManager[Union[ConversationTask, str, Dict[str, Any]]]()

        # Create parser refiner with the right queue manager type
        parser_refiner = ParserRefiner(db_manager, queue_manager_for_parser)

        # Create conversation worker
        worker = ConversationWorker(
            parser_refiner=parser_refiner,
            queue_manager=queue_manager,
            conversation_manager=conversation_manager,
            db_manager=db_manager,
            emotion_manager=emotion_manager,
            poll_interval=1.0,  # Short for demonstration
            processing_timeout=10.0,
            batch_size=5,
            enable_logging=True,
        )

        logger.info("Conversation worker initialized")

        # Start the worker
        worker.start()
        logger.info("Worker thread started")

        # Add some sample conversations
        sample_conversations = [
            [
                ("user", "Tell me about algorithms"),
                ("user", "How are algorithms used in AI?"),
                ("user", "Thanks for the explanation"),
            ],
            [
                ("user", "What is machine learning?"),
                ("user", "Can you explain neural networks?"),
            ],
        ]

        # Submit sample conversations
        for conversation in sample_conversations:
            # Create new conversation for each sample
            conversation_id = conversation_manager.start_conversation()
            logger.info(f"Created conversation {conversation_id}")

            for speaker, message in conversation:
                # Create task
                task: ConversationTask = {
                    "task_id": f"task_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
                    "conversation_id": conversation_id,
                    "message": message,
                    "speaker": speaker,
                    "timestamp": time.time(),
                    "priority": 1,
                    "context": {"generate_response": True},
                }

                # Enqueue task
                queue_manager.enqueue(task)
                logger.info(f"Enqueued message: '{message}'")

                # Small delay between messages
                time.sleep(0.5)

        # Monitor worker status
        for _ in range(5):
            status = worker.get_status()
            logger.info(f"Worker status: {status['state']}")
            logger.info(f"Queue size: {status['queue_size']}")
            logger.info(f"Processed: {status['processed_count']}")
            logger.info(f"Success: {status['success_count']}")
            logger.info(f"Errors: {status['error_count']}")

            if status["recent_errors"]:
                logger.info(f"Recent errors: {status['recent_errors']}")

            # Wait for processing
            time.sleep(2.0)

        # Demonstrate pause/resume
        logger.info("Pausing worker...")
        worker.pause()
        time.sleep(2.0)

        status = worker.get_status()
        logger.info(f"Worker status while paused: {status['state']}")

        logger.info("Resuming worker...")
        worker.resume()
        time.sleep(2.0)

        status = worker.get_status()
        logger.info(f"Worker status after resume: {status['state']}")

        # Final status before shutdown
        logger.info("Final worker status:")
        final_status = worker.get_status()
        for key, value in final_status.items():
            if key != "conversation_metrics":
                logger.info(f"  {key}: {value}")

        # Show conversation metrics
        logger.info("Conversation metrics:")
        for key, value in final_status["conversation_metrics"].items():
            logger.info(f"  {key}: {value}")

        # Stop the worker
        logger.info("Stopping worker...")
        worker.stop()
        worker.join(timeout=5.0)  # Wait for thread to terminate

        logger.info("Worker shutdown complete")

    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}", exc_info=True)

    finally:
        # Ensure worker is stopped
        if worker and worker.is_alive():
            worker.stop()
            worker.join(timeout=5.0)


if __name__ == "__main__":
    main()
