
from typing import Dict, List


class ConversationMemory:
    """Store conversation history separately for each session."""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.sessions: Dict[str, List[dict]] = {}

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id cannot be empty")

        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")

        if not content.strip():
            raise ValueError("content cannot be empty")

        history = self.sessions.setdefault(session_id, [])
        history.append({"role": role, "content": content})
        self.sessions[session_id] = history[-self.max_messages:]

    def get_history(self, session_id: str) -> List[dict]:
        return self.sessions.get(session_id, []).copy()

    def clear_history(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
    
    def get_context(self, session_id: str) -> str:
        """Return conversation history as readable text."""
        history = self.get_history(session_id)

        return "\n".join(
            f'{message["role"]}: {message["content"]}'
            for message in history
        )



if __name__ == "__main__":
    memory = ConversationMemory(max_messages=5)

    memory.add_message(
        "customer_001",
        "user",
        "Where is my order?",
    )
    memory.add_message(
        "customer_001",
        "assistant",
        "Please check your order tracking details.",
    )
    memory.add_message(
        "customer_001",
        "user",
        "How long will delivery take?",
    )

    print("Conversation history:")
    for message in memory.get_history("customer_001"):
        print(f'{message["role"]}: {message["content"]}')

    print("\nMemory test completed successfully.")
