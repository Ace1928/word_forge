"""
Demonstration of ConversationManager functionality with multi-model response generation.
"""

import logging
import time
from pathlib import Path

from word_forge.conversation.conversation_manager import ConversationManager
from word_forge.conversation.conversation_models import (
    EidosianIdentityModel,  # Import the real model
    MockAffectiveLexicalModel,
    MockLightweightModel,
)
from word_forge.database.database_manager import DBManager
from word_forge.demos.vector_worker_demo import temporary_database
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.vectorizer.vector_store import StorageType, VectorStore


def main() -> None:
    """
    Demonstrate the usage of ConversationManager with multi-model response generation.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ConversationDemo")

    temp_db_path = Path("./temp_conv_demo_db.sqlite")

    try:
        # Use temporary database context
        with temporary_database(temp_db_path) as db_path:
            logger.info(f"Using temporary database at {db_path}")
            db_manager = DBManager(db_path=str(db_path))
            db_manager.create_tables()

            # Initialize required managers
            emotion_manager = EmotionManager(db_manager)
            graph_manager = GraphManager(db_manager)
            graph_manager.ensure_sample_data()
            graph_manager.build_graph()
            vector_store = VectorStore(
                storage_type=StorageType.MEMORY,
                db_manager=db_manager,
                emotion_manager=emotion_manager,
            )

            # Initialize Models (Use the real Identity Model)
            lightweight_model = MockLightweightModel()
            affective_model = MockAffectiveLexicalModel()
            identity_model = (
                EidosianIdentityModel()
            )  # Instantiate the real identity model

            # Initialize conversation manager with all components
            conversation_manager = ConversationManager(
                db_manager=db_manager,
                emotion_manager=emotion_manager,
                graph_manager=graph_manager,
                vector_store=vector_store,
                lightweight_model=lightweight_model,
                affective_model=affective_model,
                identity_model=identity_model,  # Pass the real identity model
            )
            logger.info(
                "Conversation manager initialized with Eidosian Identity Model."
            )

            # Start a new conversation
            conversation_id = conversation_manager.start_conversation()
            logger.info(f"Started new conversation with ID: {conversation_id}")

            # Add user message and trigger response generation
            user_message = "Hello! Tell me about recursion."
            logger.info("\n=== Adding User Message & Generating Response ===")
            logger.info(f"User: {user_message}")
            message_id = conversation_manager.add_message(
                conversation_id, "User", user_message, generate_response=True
            )
            logger.info(f"Added user message (ID: {message_id})")

            # Add another user message
            user_message_2 = "That sounds complex but interesting."
            logger.info("\n=== Adding Second User Message & Generating Response ===")
            logger.info(f"User: {user_message_2}")
            message_id_2 = conversation_manager.add_message(
                conversation_id, "User", user_message_2, generate_response=True
            )
            logger.info(f"Added second user message (ID: {message_id_2})")

            # Retrieve and display the conversation
            logger.info("\n=== Final Conversation Transcript ===")
            conversation = conversation_manager.get_conversation(conversation_id)

            print(f"Conversation ID: {conversation['id']}")
            print(f"Status: {conversation['status']}")
            print(f"Created: {time.ctime(conversation['created_at'])}")
            print(f"Last updated: {time.ctime(conversation['updated_at'])}")
            print(f"Message count: {len(conversation['messages'])}")

            print("\n--- Messages ---")
            for msg in conversation["messages"]:
                print(f"\n{msg['speaker']} ({time.ctime(msg['timestamp'])}):")
                print(f"  \"{msg['text']}\"")
                if msg["emotion"]:
                    emotion = msg["emotion"]["emotion_label"]
                    confidence = msg["emotion"]["confidence"]
                    print(f"  Emotion: {emotion} (confidence: {confidence:.2f})")

            # End the conversation
            conversation_manager.end_conversation(conversation_id)
            logger.info("\nConversation marked as COMPLETED")

            # Verify the status change
            updated_conversation = conversation_manager.get_conversation(
                conversation_id
            )
            logger.info(f"Final status: {updated_conversation['status']}")

    except Exception as e:
        logger.error(f"Conversation demo failed: {e}", exc_info=True)
    finally:
        logger.info("Demo finished.")


if __name__ == "__main__":
    main()
