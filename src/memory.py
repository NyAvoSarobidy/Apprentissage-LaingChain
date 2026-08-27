
# memory.py — Gestion de l'historique conversationnel

from langchain_core.messages import HumanMessage, AIMessage


def init_memory(session_state):
    if "messages" not in session_state:
        session_state.messages = []


def add_message(session_state, role: str, content: str):
    session_state.messages.append({"role": role, "content": content})


def get_langchain_messages(session_state) -> list:
    messages = []
    for msg in session_state.messages:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def get_conversation_history(session_state) -> str:
    if not session_state.messages:
        return ""

    lines = []
    for msg in session_state.messages:
        role_label = "Utilisateur" if msg["role"] == "human" else "Assistant"
        lines.append(f"{role_label}: {msg['content']}")

    return "\n".join(lines)


def clear_memory(session_state):
    session_state.messages = []
