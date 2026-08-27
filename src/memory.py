"""
memory.py — Gestion de l'historique conversationnel
=====================================================
Gère la sauvegarde et la récupération des messages de conversation.

Dans Streamlit, le script est relancé à chaque interaction.
On utilise donc st.session_state pour persister l'historique
entre les re-exécutions.

Structure de l'historique :
    messages = [
        {"role": "human", "content": "Question 1"},
        {"role": "ai", "content": "Réponse 1"},
        {"role": "human", "content": "Question 2"},
        ...
    ]
"""

from langchain_core.messages import HumanMessage, AIMessage


def init_memory(session_state):
    """
    Initialise l'historique dans session_state si inexistant.

    Streamlit ne persiste pas les données entre les re-exécutions
    du script. session_state est le mécanisme officiel pour
    conserver des données (comme l'historique de chat).
    """
    if "messages" not in session_state:
        session_state.messages = []


def add_message(session_state, role: str, content: str):
    """
    Ajoute un message à l'historique.

    Args:
        session_state: l'état Streamlit
        role: "human" ou "ai"
        content: le texte du message
    """
    session_state.messages.append({"role": role, "content": content})


def get_langchain_messages(session_state) -> list:
    """
    Convertit l'historique en messages LangChain.

    LangChain attend des objets HumanMessage / AIMessage
    pour construire le contexte de conversation.

    Retourne une liste de messages LangChain.
    """
    messages = []
    for msg in session_state.messages:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def get_conversation_history(session_state) -> str:
    """
    Retourne l'historique formaté en texte simple.

    Utile pour passer l'historique dans un prompt
    (approche naïve de la mémoire).
    """
    if not session_state.messages:
        return ""

    lines = []
    for msg in session_state.messages:
        role_label = "Utilisateur" if msg["role"] == "human" else "Assistant"
        lines.append(f"{role_label}: {msg['content']}")

    return "\n".join(lines)


def clear_memory(session_state):
    """
    Efface l'historique de conversation.
    """
    session_state.messages = []
