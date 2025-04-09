"""
Interactive demonstration of ConversationManager functionality with multi-model response generation.

Allows real-time interaction with the Word Forge conversation system, showcasing
the multi-stage response pipeline including Reflexive, Lightweight, Affective/Lexical,
and Eidosian Identity models.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from word_forge.conversation.conversation_manager import ConversationManager
from word_forge.conversation.conversation_models import (
    EidosianIdentityModel,
    MockAffectiveLexicalModel,
    MockLightweightModel,
    MockReflexiveModel,
)
from word_forge.conversation.conversation_types import ConversationDict
from word_forge.database.database_manager import DBManager
from word_forge.demos.vector_worker_demo import temporary_database
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.vectorizer.vector_store import StorageType, VectorStore


def display_conversation(conversation: Optional[ConversationDict]) -> None:
    """
    Helper function to display conversation messages neatly.

    Formats and prints the conversation transcript, including message timestamps,
    speakers, text, and detected emotions with confidence scores.

    Args:
        conversation: The conversation data dictionary, or None if not found.
    """
    if not conversation:
        print("\n--- Conversation Not Found ---")
        return

    print("\n--- Conversation Transcript ---")
    print(f"ID: {conversation['id']}, Status: {conversation['status']}")
    if not conversation["messages"]:
        print("  (No messages yet)")
    else:
        for msg in conversation["messages"]:
            timestamp = msg.get("timestamp", 0.0)
            if not isinstance(timestamp, (int, float)):
                timestamp = 0.0

            timestamp_str = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(timestamp)
            )
            emotion_str = ""
            emotion_data = msg.get("emotion")
            if isinstance(emotion_data, dict):
                label = emotion_data.get("emotion_label")
                confidence = emotion_data.get("confidence")
                if label is not None and confidence is not None:
                    try:
                        confidence_float = float(confidence)
                        emotion_str = f" (Emotion: {label} [{confidence_float:.2f}])"
                    except (ValueError, TypeError):
                        emotion_str = f" (Emotion: {label} [Invalid Confidence])"
                elif label is not None:
                    emotion_str = f" (Emotion: {label} [Confidence Missing])"
                else:
                    emotion_str = " (Emotion: Data incomplete)"

            print(
                f"[{timestamp_str}] {msg.get('speaker', 'Unknown')}: {msg.get('text', '')}{emotion_str}"
            )
    print("--- End Transcript ---")


def main() -> None:
    """
    Runs an interactive command-line demo for the ConversationManager.

    Sets up a temporary database, initializes all required managers and models
    (including Reflexive, Lightweight, Affective/Lexical, and Identity),
    starts a conversation, and enters a loop allowing the user to chat with
    the assistant until 'quit' is entered.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("InteractiveConversationDemo")

    temp_db_path = Path("./temp_interactive_conv_demo_db.sqlite")

    try:
        # Use temporary database context
        with temporary_database(temp_db_path) as db_path:
            logger.info(f"Using temporary database at {db_path}")
            db_manager = DBManager(db_path=str(db_path))

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
            logger.info("Initialized DB, Emotion, Graph, and Vector managers.")

            # Initialize Models (including Reflexive and actual Identity)
            reflexive_model = MockReflexiveModel()
            lightweight_model = MockLightweightModel()
            affective_model = MockAffectiveLexicalModel()
            identity_model = EidosianIdentityModel()
            logger.info(
                "Initialized conversation models (Reflexive, Lightweight, Affective, EidosianIdentity)."
            )

            # Initialize conversation manager with all models
            conversation_manager = ConversationManager(
                db_manager=db_manager,
                emotion_manager=emotion_manager,
                graph_manager=graph_manager,
                vector_store=vector_store,
                reflexive_model=reflexive_model,
                lightweight_model=lightweight_model,
                affective_model=affective_model,
                identity_model=identity_model,
            )
            logger.info("Conversation manager initialized.")

            # Start a new conversation
            try:
                conversation_id = conversation_manager.start_conversation()
                logger.info(f"Started new conversation with ID: {conversation_id}")
                print(
                    f"\nAssistant: Hello! I'm ready to chat. (Conversation ID: {conversation_id}). Type 'quit' to exit."
                )
            except Exception as e:
                logger.error(f"Failed to start conversation: {e}", exc_info=True)
                print(f"Error: Could not start conversation: {e}")
                return

            # Interactive loop
            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input.lower() == "quit":
                        print("Assistant: Goodbye!")
                        break

                    if not user_input:
                        continue

                    print("Assistant thinking...")
                    try:
                        message_id = conversation_manager.add_message(
                            conversation_id, "User", user_input, generate_response=True
                        )
                        logger.info(
                            f"User message {message_id} added and response triggered."
                        )

                        conversation = conversation_manager.get_conversation_if_exists(
                            conversation_id
                        )
                        display_conversation(conversation)

                    except Exception as add_msg_e:
                        logger.error(
                            f"Error adding message or generating response: {add_msg_e}",
                            exc_info=True,
                        )
                        print(
                            f"Assistant: Sorry, I encountered an error processing that: {add_msg_e}"
                        )
                        conversation = conversation_manager.get_conversation_if_exists(
                            conversation_id
                        )
                        display_conversation(conversation)

                except EOFError:
                    print("\nAssistant: Goodbye!")
                    break
                except KeyboardInterrupt:
                    print("\nAssistant: Exiting conversation. Goodbye!")
                    break
                except Exception as loop_e:
                    logger.error(
                        f"Error during interaction loop: {loop_e}", exc_info=True
                    )
                    print(f"Assistant: An unexpected error occurred: {loop_e}")

            try:
                conversation_manager.end_conversation(conversation_id)
                logger.info(f"Conversation {conversation_id} marked as COMPLETED")
            except Exception as end_e:
                logger.error(
                    f"Failed to mark conversation {conversation_id} as completed: {end_e}",
                    exc_info=True,
                )

    except Exception as e:
        logger.error(f"Conversation demo failed critically: {e}", exc_info=True)
        print(f"Critical error running demo: {e}")
    finally:
        if temp_db_path.exists():
            try:
                temp_db_path.unlink()
                logger.info(f"Removed temporary database: {temp_db_path}")
            except OSError as e:
                logger.error(f"Error removing temporary database {temp_db_path}: {e}")
        logger.info("Demo finished.")


if __name__ == "__main__":
    main()
