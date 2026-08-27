"""
test_etape3.py — Test rapide de l'Étape 3
==========================================
Vérifie la logique de reformulation contextuelle
SANS avoir besoin des clés API.

On teste :
  - La détection de questions autonomes (pas besoin de reformuler)
  - La détection de questions dépendantes (à reformuler)
  - Le format du prompt de reformulation

Usage :
    python test_etape3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_chain import REFORMULATION_PROMPT

# ---------------------------------------------------------------------------
# Test 1 : Le prompt de reformulation contient les bonnes instructions
# ---------------------------------------------------------------------------


def test_reformulation_prompt():
    """
    Vérifie que le prompt de reformulation contient les éléments clés :
      - Instruction de ne pas répondre à la question
      - Mention de l'historique
      - Format de sortie attendu
    """
    print("=" * 60)
    print("🧪 Test 1 : Contenu du prompt de reformulation")
    print("=" * 60)

    # Vérifications
    assert "reformule" in REFORMULATION_PROMPT.lower(), "Le prompt doit mentionner la reformulation"
    assert "historique" in REFORMULATION_PROMPT.lower(), "Le prompt doit mentionner l'historique"
    assert "ne réponds pas" in REFORMULATION_PROMPT.lower(), "Le prompt doit interdire de répondre"
    assert "{history}" in REFORMULATION_PROMPT, "Le prompt doit contenir le placeholder {history}"
    assert "{question}" in REFORMULATION_PROMPT, "Le prompt doit contenir le placeholder {question}"

    print("   ✅ Le prompt contient 'reformule'")
    print("   ✅ Le prompt contient 'historique'")
    print("   ✅ Le prompt interdit de répondre à la question")
    print("   ✅ Les placeholders {history} et {question} sont présents")
    print("   ✅ Test 1 réussi !\n")


# ---------------------------------------------------------------------------
# Test 2 : Détection des questions dépendantes
# ---------------------------------------------------------------------------


def test_dependent_questions():
    """
    Vérifie qu'on peut identifier les questions qui dépendent du contexte.

    Une question dépendante contient :
      - Des pronoms démonstratifs (cela, ceci, ce dernier)
      - Des références implicites (le deuxième cas, le précédent)
      - Des adverbes de liaison (ensuite, de plus, par ailleurs)
    """
    print("=" * 60)
    print("🧪 Test 2 : Détection des questions dépendantes")
    print("=" * 60)

    import re

    # Mots-clés qui indiquent une question dépendante
    # On utilise des word boundaries pour éviter les faux positifs
    # (ex: "la" article vs "la" pronom)
    DEPENDENT_PATTERNS = [
        r'\bcela\b', r'\bceci\b', r'\bce dernier\b', r'\ble deuxième\b',
        r'\ble précédent\b', r'\bensuite\b', r'\bde plus\b', r'\bpar ailleurs\b',
        r'\bet pour\b', r'\bil\b', r'\belle\b', r'\bils\b', r'\belles\b',
        r'\bceux\b', r'\bcelles\b', r'\bcelui\b', r'\bcelle\b',
        r'\bses\b', r'\bson\b', r'\bsa\b', r'\bleurs\b', r'\bleur\b',
        r"l'a\b", r"l'\b",  # pronom élidé
    ]

    def is_dependent(question: str) -> bool:
        """Détecte si une question dépend du contexte."""
        q_lower = question.lower()
        for pattern in DEPENDENT_PATTERNS:
            if re.search(pattern, q_lower):
                return True
        return False

    # Questions dépendantes (devraient être détectées)
    dependent_questions = [
        "Et pour le deuxième cas ?",
        "Quels sont ses avantages ?",
        "Comment cela fonctionne-t-il ?",
        "Qui l'a créé ?",
        "Et ensuite ?",
    ]

    # Questions autonomes (ne devraient PAS être détectées)
    autonomous_questions = [
        "Qu'est-ce que le RAG ?",
        "Comment fonctionne la recherche vectorielle ?",
        "Quels sont les types d'IA ?",
        "Qu'est-ce que ChromaDB ?",
    ]

    print("   Questions dépendantes détectées :")
    for q in dependent_questions:
        result = is_dependent(q)
        assert result, f"'{q}' devrait être détectée comme dépendante"
        print(f"      ✅ '{q}' → dépendante")

    print("\n   Questions autonomes détectées :")
    for q in autonomous_questions:
        result = is_dependent(q)
        assert not result, f"'{q}' devrait être détectée comme autonome"
        print(f"      ✅ '{q}' → autonome")

    print("\n   ✅ Test 2 réussi !\n")


# ---------------------------------------------------------------------------
# Test 3 : Construction de l'historique pour le reformulateur
# ---------------------------------------------------------------------------


def test_history_formatting():
    """
    Vérifie que l'historique est correctement formaté
    pour être injecté dans le prompt de reformulation.
    """
    print("=" * 60)
    print("🧪 Test 3 : Format de l'historique")
    print("=" * 60)

    # Simulons un historique de conversation
    history = [
        {"question": "Qu'est-ce que le RAG ?", "answer": "Le RAG combine recherche et génération."},
        {"question": "Quels sont les types d'IA ?", "answer": "IA faible et IA forte."},
    ]

    # Construire l'historique textuel (même logique que dans main)
    lines = []
    for h in history:
        lines.append(f"Utilisateur: {h['question']}")
        lines.append(f"Assistant: {h['answer']}")
    history_text = "\n".join(lines)

    print("   Historique produit :")
    for line in history_text.split("\n"):
        print(f"   | {line}")

    # Vérifications
    assert "Utilisateur: Qu'est-ce que le RAG ?" in history_text
    assert "Assistant: Le RAG combine recherche et génération." in history_text
    assert "Utilisateur: Quels sont les types d'IA ?" in history_text
    assert "Assistant: IA faible et IA forte." in history_text

    print("\n   ✅ L'historique est correctement formaté")
    print("   ✅ Test 3 réussi !\n")


# ---------------------------------------------------------------------------
# Test 4 : Le prompt reformulation avec historique injecté
# ---------------------------------------------------------------------------


def test_prompt_with_history():
    """
    Vérifie que le prompt de reformulation peut être rempli
    avec un historique et une question.
    """
    print("=" * 60)
    print("🧪 Test 4 : Prompt de reformulation rempli")
    print("=" * 60)

    history = "Utilisateur: Quels sont les types d'IA ?\nAssistant: IA faible et IA forte."
    question = "Et pour le deuxième cas ?"

    # Remplir le prompt
    filled_prompt = REFORMULATION_PROMPT.format(history=history, question=question)

    print("   Prompt rempli (extrait) :")
    for line in filled_prompt.split("\n")[:8]:
        print(f"   | {line}")
    print("   | ...")

    # Vérifications
    assert history in filled_prompt, "L'historique doit être injecté"
    assert question in filled_prompt, "La question doit être injectée"
    assert "{history}" not in filled_prompt, "Le placeholder doit être remplacé"
    assert "{question}" not in filled_prompt, "Le placeholder doit être remplacé"

    print("\n   ✅ L'historique et la question sont injectés")
    print("   ✅ Les placeholders sont remplacés")
    print("   ✅ Test 4 réussi !\n")


# ---------------------------------------------------------------------------
# Test 5 : Configuration de la reformulation
# ---------------------------------------------------------------------------


def test_reformulation_config():
    """
    Vérifie que la configuration de la reformulation est correcte.
    """
    print("=" * 60)
    print("🧪 Test 5 : Configuration")
    print("=" * 60)

    from src.rag_chain import USE_REFORMULATION, REFORMULATOR_MODEL, LLM_MODEL

    # Vérifications
    assert USE_REFORMULATION is True, "La reformulation devrait être activée par défaut"
    assert REFORMULATOR_MODEL != LLM_MODEL, "Le reformulateur devrait utiliser un modèle différent (plus rapide)"
    # On ne teste PAS un nom de fournisseur en dur ici : le projet peut basculer
    # de Gemini vers Claude ou Groq sans que ce test devienne faux.
    # La regle qui compte est structurelle : le reformulateur doit etre un modele
    # plus leger que le LLM principal. Chez tous les fournisseurs, ces modeles
    # portent un suffixe du type "lite", "mini", "haiku", "flash", "small".
    MODELES_LEGERS = ("lite", "mini", "haiku", "flash", "small", "8b", "instant")
    assert any(marqueur in REFORMULATOR_MODEL.lower() for marqueur in MODELES_LEGERS), (
        f"Le reformulateur '{REFORMULATOR_MODEL}' ne ressemble pas a un modele leger. "
        f"Attendu un nom contenant l'un de : {MODELES_LEGERS}"
    )

    print(f"   Reformulation activée : {USE_REFORMULATION}")
    print(f"   Modèle reformulateur : {REFORMULATOR_MODEL}")
    print(f"   Modèle LLM principal : {LLM_MODEL}")
    print("   ✅ Configuration correcte")
    print("   ✅ Test 5 réussi !\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("\n🧪 Tests de l'Étape 3 — Reformulation contextuelle\n")
    print("   (Tests locaux, pas de clé API requise)\n")

    try:
        test_reformulation_prompt()
        test_dependent_questions()
        test_history_formatting()
        test_prompt_with_history()
        test_reformulation_config()

        print("=" * 60)
        print("🎉 Tous les tests de l'Étape 3 sont réussis !")
        print("=" * 60)
        print()
        print("📋 Pour tester avec les API :")
        print("   1. Assure-toi d'avoir des PDF indexés")
        print("   2. python -m src.rag_chain")
        print("   3. Pose une question autonome, puis une question dépendante")
        print("      comme 'Et pour le deuxième cas ?'")
        print()

    except AssertionError as e:
        print(f"   ❌ Échec : {e}")
        exit(1)
