"""
Demonstration of ConversationManager functionality with multi-model response generation.
"""

import logging
import time
from pathlib import Path

from word_forge.conversation.conversation_manager import ConversationManager
from word_forge.conversation.conversation_models import (
    EidosianIdentityModel,
    MockAffectiveLexicalModel,
    MockLightweightModel,
)
from word_forge.conversation.conversation_types import ConversationDict
from word_forge.database.database_manager import DBManager
from word_forge.demos.vector_worker_demo import temporary_database
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.vectorizer.vector_store import StorageType, VectorStore


def main() -> None:
    """
    Demonstrate the usage of ConversationManager with multi-model response generation.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("ConversationDemo")

    temp_db_path = Path("./temp_conv_demo_db.sqlite")

    try:
        # Use temporary database context
        with temporary_database(temp_db_path) as db_path:
            logger.info(f"Using temporary database at {db_path}")
            db_manager = DBManager(db_path=str(db_path))
            db_manager.create_tables()  # Ensure core tables exist

            # Initialize required managers
            emotion_manager = EmotionManager(db_manager)
            # Ensure emotion tables exist
            emotion_manager._create_tables()  # Call internal method to ensure tables

            graph_manager = GraphManager(db_manager)
            # Ensure graph tables exist and add sample data if needed
            graph_manager.db_manager.create_tables()  # Ensure graph DB tables exist
            graph_manager.ensure_sample_data()
            graph_manager.build_graph()  # Build graph after ensuring data

            vector_store = VectorStore(
                storage_type=StorageType.MEMORY,  # Use memory for demo speed
                db_manager=db_manager,
                emotion_manager=emotion_manager,
            )
            logger.info("Initialized DB, Emotion, Graph, and Vector managers.")

            # Initialize Models
            lightweight_model = MockLightweightModel()
            affective_model = MockAffectiveLexicalModel()
            identity_model = EidosianIdentityModel()  # Use the real identity model
            logger.info(
                "Initialized conversation models (Lightweight, Affective, EidosianIdentity)."
            )

            # Initialize conversation manager with all components
            conversation_manager = ConversationManager(
                db_manager=db_manager,
                emotion_manager=emotion_manager,
                graph_manager=graph_manager,
                vector_store=vector_store,
                lightweight_model=lightweight_model,
                affective_model=affective_model,
                identity_model=identity_model,
            )
            # Ensure conversation tables exist
            conversation_manager._create_tables()
            logger.info("Conversation manager initialized with multi-model pipeline.")

            # Start a new conversation
            conversation_id = conversation_manager.start_conversation()
            logger.info(f"Started new conversation with ID: {conversation_id}")

            # Add user message and trigger response generation
            user_message = "Hello! Tell me about recursion in programming."
            logger.info("\n=== Adding User Message & Generating Response ===")
            logger.info(f"User: {user_message}")
            # Set generate_response=True to trigger the pipeline
            message_id = conversation_manager.add_message(
                conversation_id, "User", user_message, generate_response=True
            )
            logger.info(
                f"Added user message (ID: {message_id}) and triggered response generation."
            )

            # Wait briefly for response generation if it's asynchronous (adjust if needed)
            time.sleep(1)  # Give models time to process (especially if LLM is involved)

            # Add another user message
            user_message_2 = "That sounds complex but interesting. How is it different from iteration?"
            logger.info("\n=== Adding Second User Message & Generating Response ===")
            logger.info(f"User: {user_message_2}")
            message_id_2 = conversation_manager.add_message(
                conversation_id, "User", user_message_2, generate_response=True
            )
            logger.info(
                f"Added second user message (ID: {message_id_2}) and triggered response generation."
            )

            # Wait again
            time.sleep(1)

            # Retrieve and display the conversation
            logger.info("\n=== Final Conversation Transcript ===")
            # Use the specific ConversationDict type hint
            conversation: ConversationDict = conversation_manager.get_conversation(
                conversation_id
            )

            print(f"\nConversation ID: {conversation['id']}")
            print(f"Status: {conversation['status']}")
            print(f"Created: {time.ctime(conversation['created_at'])}")
            print(f"Last updated: {time.ctime(conversation['updated_at'])}")
            print(f"Message count: {len(conversation['messages'])}")

            print("\n--- Messages ---")
            # Use the specific MessageDict type hint
            for msg in conversation["messages"]:
                timestamp_str = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(msg["timestamp"])
                )
                print(f"\n{msg['speaker']} ({timestamp_str}):")
                print(f"  \"{msg['text']}\"")
                # Check if emotion data exists and is not None
                if msg.get("emotion"):
                    emotion_info = msg["emotion"]
                    # Check if the required keys exist in the emotion dict
                    if (
                        emotion_info
                        and "emotion_label" in emotion_info
                        and "confidence" in emotion_info
                    ):
                        emotion = emotion_info["emotion_label"]
                        confidence = emotion_info["confidence"]
                        print(f"  Emotion: {emotion} (confidence: {confidence:.2f})")
                    else:
                        print("  Emotion: (Data incomplete)")
                else:
                    print("  Emotion: (Not analyzed)")

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
        # Cleanup: Remove the temporary database file if it exists
        if temp_db_path.exists():
            try:
                temp_db_path.unlink()
                logger.info(f"Removed temporary database: {temp_db_path}")
            except OSError as e:
                logger.error(f"Error removing temporary database {temp_db_path}: {e}")
        logger.info("Demo finished.")


if __name__ == "__main__":
    main()
