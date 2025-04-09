"""
Demonstration of ConversationManager functionality.
"""

import time

from word_forge.conversation.conversation_manager import (
    ConversationManager,
    ConversationNotFoundError,
)
from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager


def main() -> None:
    """
    Demonstrate the usage of ConversationManager with a complete workflow example.
    """

    # Initialize database manager
    db_path = "word_forge.sqlite"
    db_manager = DBManager(db_path)
    print(f"Using database at {db_path}")

    # Initialize emotion manager for sentiment analysis
    emotion_manager = EmotionManager(db_manager)

    # Initialize conversation manager with emotion integration
    conversation_manager = ConversationManager(db_manager, emotion_manager)
    print("Conversation manager initialized with emotion analysis capabilities")

    # Start a new conversation
    conversation_id = conversation_manager.start_conversation()
    print(f"Started new conversation with ID: {conversation_id}")

    # Add messages to the conversation
    messages = [
        ("User", "Hello! Can you help me understand what machine learning is?"),
        (
            "Assistant",
            "Of course! Machine learning is a branch of artificial intelligence focused on building systems that learn from data.",
        ),
        ("User", "That sounds interesting! Can you give me a simple example?"),
        (
            "Assistant",
            "Sure! A common example is email spam detection. The system learns patterns from emails marked as spam to identify new spam messages.",
        ),
        ("User", "Thanks, that makes sense! I appreciate your help."),
        (
            "Assistant",
            "You're welcome! I'm glad I could help. Feel free to ask if you have more questions.",
        ),
    ]

    print("\n=== Adding messages to conversation ===")
    for speaker, text in messages:
        message_id = conversation_manager.add_message(conversation_id, speaker, text)
        print(f"Added message from {speaker} (ID: {message_id})")

    # Retrieve and display the conversation
    print("\n=== Conversation Transcript ===")
    conversation = conversation_manager.get_conversation(conversation_id)

    print(f"Conversation ID: {conversation['id']}")
    print(f"Status: {conversation['status']}")
    print(f"Created: {time.ctime(conversation['created_at'])}")
    print(f"Last updated: {time.ctime(conversation['updated_at'])}")
    print(f"Message count: {len(conversation['messages'])}")

    print("\n=== Messages with Emotion Analysis ===")
    for msg in conversation["messages"]:
        print(f"\n{msg['speaker']} ({time.ctime(msg['timestamp'])}):")
        print(f"  \"{msg['text']}\"")

        if msg["emotion"]:
            emotion = msg["emotion"]["emotion_label"]
            confidence = msg["emotion"]["confidence"]
            print(f"  Emotion: {emotion} (confidence: {confidence:.2f})")

    # End the conversation
    conversation_manager.end_conversation(conversation_id)
    print("\nConversation marked as COMPLETED")

    # Verify the status change
    updated_conversation = conversation_manager.get_conversation(conversation_id)
    print(f"Final status: {updated_conversation['status']}")

    # Demonstrate error handling with a non-existent conversation
    non_existent_id = 9999
    print(f"\nTrying to retrieve non-existent conversation (ID: {non_existent_id})...")
    try:
        conversation_manager.get_conversation(non_existent_id)
    except ConversationNotFoundError as e:
        print(f"Error (as expected): {e}")

    # Use the safe lookup method
    result = conversation_manager.get_conversation_if_exists(non_existent_id)
    print(f"Safe lookup result: {'Found' if result else 'Not found'}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
