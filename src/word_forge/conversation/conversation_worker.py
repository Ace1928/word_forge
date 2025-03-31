import threading
import time
import traceback

from word_forge.conversation.conversation_manager import ConversationManager
from word_forge.parser.parser_refiner import ParserRefiner
from word_forge.queue.queue_manager import QueueManager


class ConversationWorker(threading.Thread):
    """
    Processes conversation tasks in the background (e.g., generating replies).
    For each queued item, it retrieves conversation context and uses the ParserRefiner
    or other LLM logic to generate a response, then saves it.
    """

    def __init__(
        self,
        conversation_manager: ConversationManager,
        parser_refiner: ParserRefiner,
        queue_manager: QueueManager,
        sleep_interval: float = 1.0,
        daemon: bool = True,
    ):
        super().__init__(daemon=daemon)
        self.conversation_manager = conversation_manager
        self.parser_refiner = parser_refiner
        self.queue_manager = queue_manager
        self.sleep_interval = sleep_interval
        self._stop_flag = False

    def run(self):
        while not self._stop_flag:
            # We'll store tasks in the queue as (task_type, conversation_id, message_text)
            task = self.queue_manager.dequeue_word()
            if task:
                if isinstance(task, tuple) and len(task) == 3:
                    task_type, conv_id, message_text = task
                    if task_type == "conversation_message":
                        self._process_conversation_message(conv_id, message_text)
                else:
                    # If the queue item is not recognized, skip
                    pass
            else:
                time.sleep(self.sleep_interval)

    def stop(self):
        self._stop_flag = True

    def _process_conversation_message(self, conversation_id: int, user_text: str):
        """
        Called when we have a new user message.
        1. Add the user's message.
        2. Generate a system reply using the parser or custom LLM logic.
        3. Save the system's reply back to the conversation.
        """
        try:
            self.conversation_manager.add_message(conversation_id, "USER", user_text)
            # Retrieve entire conversation for context
            conv_data = self.conversation_manager.get_conversation(conversation_id)
            if not conv_data:
                return

            # Very naive approach: ask parser_refiner to 'process_word' or do custom logic
            # For a real chat, you'd have a dedicated LLM or conversation logic:
            system_reply = self._generate_reply(user_text)

            # Save system reply
            self.conversation_manager.add_message(
                conversation_id, "SYSTEM", system_reply
            )

        except Exception as e:
            print(f"[ConversationWorker] Error: {str(e)}")
            traceback.print_exc()

    def _generate_reply(self, user_text: str) -> str:
        """
        Minimal placeholder that calls parser_refiner, or returns a witty response.
        """
        # For a more advanced approach, integrate a conversation LLM here
        if "hello" in user_text.lower():
            return "Hello! How can I help you today?"
        else:
            # Possibly do some lexical expansions or synonyms
            # For now, let's just echo or do something silly
            return f"I heard you say: '{user_text}'. That's quite interesting!"
