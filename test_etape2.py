
# test_etape2.py — Test rapide de l'Étape 2


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.memory import (
    init_memory,
    add_message,
    get_conversation_history,
    get_langchain_messages,
    clear_memory,
)


# ---------------------------------------------------------------------------
# Simulé de session_state (un simple dictionnaire)
# ---------------------------------------------------------------------------


class FakeSessionState:
    """Simule st.session_state pour les tests."""
    def __init__(self):
        self.data = {}

    def __getattr__(self, key):
        return self.data.get(key)

    def __setattr__(self, key, value):
        if key == "data":
            super().__setattr__(key, value)
        else:
            self.data[key] = value

    def __contains__(self, key):
        return key in self.data


# ---------------------------------------------------------------------------
# Test 1 : Initialisation de la mémoire
# ---------------------------------------------------------------------------


def test_init_memory():
    print("=" * 60)
    print("Test 1 : Initialisation de la mémoire")
    print("=" * 60)

    state = FakeSessionState()

    # Avant init : pas de messages
    assert "messages" not in state.data, "messages ne devrait pas exister avant init"

    init_memory(state)

    # Après init : liste vide
    assert hasattr(state, "messages"), "messages devrait exister après init"
    assert state.messages == [], "messages devrait être une liste vide"

    print("Mémoire initialisée avec une liste vide")
    print("Test 1 réussi !\n")


# ---------------------------------------------------------------------------
# Test 2 : Ajout de messages
# ---------------------------------------------------------------------------


def test_add_message():
    print("=" * 60)
    print("Test 2 : Ajout de messages")
    print("=" * 60)

    state = FakeSessionState()
    init_memory(state)

    add_message(state, "human", "Qu'est-ce que le RAG ?")
    add_message(state, "ai", "Le RAG est une technique...")
    add_message(state, "human", "Comment ça marche ?")

    assert len(state.messages) == 3, f"3 messages attendus, {len(state.messages)} trouvés"
    assert state.messages[0]["role"] == "human"
    assert state.messages[0]["content"] == "Qu'est-ce que le RAG ?"
    assert state.messages[1]["role"] == "ai"
    assert state.messages[2]["content"] == "Comment ça marche ?"

    print(f" {len(state.messages)} messages ajoutés correctement")
    print("  Test 2 réussi !\n")


# ---------------------------------------------------------------------------
# Test 3 : Historique formaté en texte
# ---------------------------------------------------------------------------


def test_get_conversation_history():
    print("=" * 60)
    print("Test 3 : Historique formaté")
    print("=" * 60)

    state = FakeSessionState()
    init_memory(state)

    add_message(state, "human", "Qu'est-ce que le RAG ?")
    add_message(state, "ai", "Le RAG combine recherche et génération.")

    history = get_conversation_history(state)

    print("   Historique produit :")
    for line in history.split("\n"):
        print(f"   | {line}")

    assert "Utilisateur: Qu'est-ce que le RAG ?" in history
    assert "Assistant: Le RAG combine recherche et génération." in history

    print("   Historique formaté correctement")
    print("   Test 3 réussi !\n")


# ---------------------------------------------------------------------------
# Test 4 : Conversion en messages LangChain
# ---------------------------------------------------------------------------


def test_get_langchain_messages():
    print("=" * 60)
    print("Test 4 : Conversion en messages LangChain")
    print("=" * 60)

    state = FakeSessionState()
    init_memory(state)

    add_message(state, "human", "Question test")
    add_message(state, "ai", "Réponse test")

    messages = get_langchain_messages(state)

    assert len(messages) == 2, f"2 messages attendus, {len(messages)} trouvés"

    from langchain_core.messages import HumanMessage, AIMessage
    assert isinstance(messages[0], HumanMessage), "Premier devrait être HumanMessage"
    assert isinstance(messages[1], AIMessage), "Deuxième devrait être AIMessage"
    assert messages[0].content == "Question test"
    assert messages[1].content == "Réponse test"

    print(f"  {len(messages)} messages LangChain créés")
    print("   Types : HumanMessage, AIMessage")
    print("   Test 4 réussi !\n")


# ---------------------------------------------------------------------------
# Test 5 : Effacement de la mémoire
# ---------------------------------------------------------------------------


def test_clear_memory():
    print("=" * 60)
    print("Test 5 : Effacement de la mémoire")
    print("=" * 60)

    state = FakeSessionState()
    init_memory(state)

    add_message(state, "human", "Question")
    add_message(state, "ai", "Réponse")
    assert len(state.messages) == 2

    clear_memory(state)
    assert len(state.messages) == 0, "La mémoire devrait être vide après clear"

    print("    Mémoire effacée correctement")
    print("    Test 5 réussi !\n")


# ---------------------------------------------------------------------------
# Test 6 : Historique vide
# ---------------------------------------------------------------------------


def test_empty_history():
    print("=" * 60)
    print(" Test 6 : Historique vide")
    print("=" * 60)

    state = FakeSessionState()
    init_memory(state)

    history = get_conversation_history(state)
    assert history == "", "L'historique vide devrait retourner une chaîne vide"

    messages = get_langchain_messages(state)
    assert messages == [], "La liste de messages devrait être vide"

    print("    Historique vide géré correctement")
    print("    Test 6 réussi !\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("\n Tests de l'Étape 2 — Mémoire conversationnelle\n")
    print("   (Tests locaux, pas de clé API requise)\n")

    try:
        test_init_memory()
        test_add_message()
        test_get_conversation_history()
        test_get_langchain_messages()
        test_clear_memory()
        test_empty_history()

        print("=" * 60)
        print(" Tous les tests de l'Étape 2 sont réussis !")
        print("=" * 60)
        print()
        print(" Pour tester l'interface Streamlit :")
        print("   1. pip install streamlit")
        print("   2. streamlit run src/app.py")
        print()

    except AssertionError as e:
        print(f"    Échec : {e}")
        exit(1)
