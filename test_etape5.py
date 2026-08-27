"""
test_etape5.py — Test rapide de l'Étape 5
==========================================
Vérifie les fonctions d'évaluation
SANS avoir besoin des clés API.

On teste :
  - L'extraction des citations
  - Le calcul de la pertinence
  - La détection des refus
  - Le chargement des questions

Usage :
    python test_etape5.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval.evaluate import (
    evaluate_citations,
    evaluate_relevance,
    evaluate_refusal,
    evaluate_chunk_retrieval,
    load_questions,
)
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Test 1 : Extraction des citations
# ---------------------------------------------------------------------------


def test_evaluate_citations():
    """
    Vérifie que l'évaluation des citations fonctionne correctement.
    """
    print("=" * 60)
    print("🧪 Test 1 : Évaluation des citations")
    print("=" * 60)

    # Réponse avec citations valides
    response = """Le RAG combine recherche et génération [cours_rag.pdf, page 1].
    Il utilise des embeddings [cours_embeddings.pdf, page 2]."""

    expected_sources = ["cours_rag.pdf", "cours_embeddings.pdf"]
    result = evaluate_citations(response, expected_sources)

    print(f"   Réponse : '{response[:60]}...'")
    print(f"   Sources attendues : {expected_sources}")
    print(f"   Score : {result['score']:.0%}")
    print(f"   Citations valides : {result['valid']}")
    print(f"   Citations invalides : {result['invalid']}")

    # Vérifications
    assert result["score"] == 1.0, "Toutes les citations devraient être valides"
    assert len(result["valid"]) == 2, "2 citations valides attendues"
    assert len(result["invalid"]) == 0, "Aucune citation invalide"

    print("   ✅ Citations valides détectées")
    print("   ✅ Test 1 réussi !\n")


# ---------------------------------------------------------------------------
# Test 2 : Citations invalides
# ---------------------------------------------------------------------------


def test_evaluate_invalid_citations():
    """
    Vérifie la détection de citations invalides (sources inventées).
    """
    print("=" * 60)
    print("🧪 Test 2 : Citations invalides")
    print("=" * 60)

    # Réponse avec une citation inventée
    response = """Le RAG combine recherche et génération [cours_rag.pdf, page 1].
    C'est inventé [faux_document.pdf, page 99]."""

    expected_sources = ["cours_rag.pdf"]
    result = evaluate_citations(response, expected_sources)

    print(f"   Réponse : '{response[:60]}...'")
    print(f"   Sources attendues : {expected_sources}")
    print(f"   Score : {result['score']:.0%}")
    print(f"   Citations valides : {result['valid']}")
    print(f"   Citations invalides : {result['invalid']}")

    # Vérifications
    assert result["score"] == 0.5, "50% des citations sont valides"
    assert len(result["valid"]) == 1, "1 citation valide"
    assert len(result["invalid"]) == 1, "1 citation invalide"

    print("   ✅ Citations invalides détectées")
    print("   ✅ Test 2 réussi !\n")


# ---------------------------------------------------------------------------
# Test 3 : Pertinence de la réponse
# ---------------------------------------------------------------------------


def test_evaluate_relevance():
    """
    Vérifie que la pertinence est correctement calculée.
    """
    print("=" * 60)
    print("🧪 Test 3 : Pertinence de la réponse")
    print("=" * 60)

    # Réponse contenant les mots-clés attendus
    response = "Le RAG combine recherche documentaire et génération de texte avec des embeddings."
    keywords = ["RAG", "recherche", "génération", "embeddings"]

    result = evaluate_relevance(response, keywords)

    print(f"   Réponse : '{response}'")
    print(f"   Mots-clés attendus : {keywords}")
    print(f"   Score : {result['score']:.0%}")
    print(f"   Mots-clés trouvés : {result['found']}")
    print(f"   Mots-clés manquants : {result['missing']}")

    # Vérifications
    assert result["score"] == 1.0, "Tous les mots-clés sont présents"
    assert len(result["found"]) == 4, "4 mots-clés trouvés"
    assert len(result["missing"]) == 0, "Aucun mot-clé manquant"

    print("   ✅ Pertinence correcte")
    print("   ✅ Test 3 réussi !\n")


# ---------------------------------------------------------------------------
# Test 4 : Pertinence partielle
# ---------------------------------------------------------------------------


def test_evaluate_partial_relevance():
    """
    Vérifie la pertinence quand seuls certains mots-clés sont présents.
    """
    print("=" * 60)
    print("🧪 Test 4 : Pertinence partielle")
    print("=" * 60)

    # Réponse incomplète
    response = "Le RAG combine recherche et génération."
    keywords = ["RAG", "recherche", "génération", "embeddings", "vecteur"]

    result = evaluate_relevance(response, keywords)

    print(f"   Réponse : '{response}'")
    print(f"   Mots-clés attendus : {keywords}")
    print(f"   Score : {result['score']:.0%}")
    print(f"   Mots-clés trouvés : {result['found']}")
    print(f"   Mots-clés manquants : {result['missing']}")

    # Vérifications
    assert result["score"] == 0.6, "3/5 mots-clés = 60%"
    assert len(result["found"]) == 3, "3 mots-clés trouvés"
    assert len(result["missing"]) == 2, "2 mots-clés manquants"

    print("   ✅ Pertinence partielle correcte")
    print("   ✅ Test 4 réussi !\n")


# ---------------------------------------------------------------------------
# Test 5 : Détection des refus
# ---------------------------------------------------------------------------


def test_evaluate_refusal():
    """
    Vérifie que les refus sont correctement détectés.
    """
    print("=" * 60)
    print("🧪 Test 5 : Détection des refus")
    print("=" * 60)

    # Cas 1 : le système devrait répondre mais refuse (incorrect)
    response_refuse = "Je ne trouve pas cette information dans les documents fournis."
    result = evaluate_refusal(response_refuse, should_answer=True)

    print(f"   Réponse : '{response_refuse}'")
    print(f"   Devrait répondre : oui")
    print(f"   Refusal correct : {result['correct']}")

    assert result["correct"] is False, "Le système refuse alors qu'il devrait répondre"
    assert result["score"] == 0.0, "Score 0 pour refus incorrect"

    # Cas 2 : le système devrait refuser et refuse (correct)
    result = evaluate_refusal(response_refuse, should_answer=False)

    print(f"\n   Réponse : '{response_refuse}'")
    print(f"   Devrait répondre : non")
    print(f"   Refusal correct : {result['correct']}")

    assert result["correct"] is True, "Le système refuse correctement"
    assert result["score"] == 1.0, "Score 1 pour refus correct"

    # Cas 3 : le système devrait répondre et répond (correct)
    response_answer = "Le RAG combine recherche et génération [cours_rag.pdf, page 1]."
    result = evaluate_refusal(response_answer, should_answer=True)

    print(f"\n   Réponse : '{response_answer[:50]}...'")
    print(f"   Devrait répondre : oui")
    print(f"   Refusal correct : {result['correct']}")

    assert result["correct"] is True, "Le système répond correctement"
    assert result["score"] == 1.0, "Score 1 pour réponse correcte"

    print("   ✅ Refusals correctement détectés")
    print("   ✅ Test 5 réussi !\n")


# ---------------------------------------------------------------------------
# Test 6 : Rappel des chunks
# ---------------------------------------------------------------------------


def test_evaluate_chunk_retrieval():
    """
    Vérifie que le rappel des chunks est correctement calculé.
    """
    print("=" * 60)
    print("🧪 Test 6 : Rappel des chunks")
    print("=" * 60)

    # Simulons des chunks récupérés
    retrieved_docs = [
        Document(page_content="...", metadata={"source": "cours_rag.pdf", "page": 0}),
        Document(page_content="...", metadata={"source": "cours_embeddings.pdf", "page": 1}),
    ]

    # Cas 1 : toutes les sources attendues sont trouvées
    expected_sources = ["cours_rag.pdf", "cours_embeddings.pdf"]
    result = evaluate_chunk_retrieval(retrieved_docs, expected_sources)

    print(f"   Sources attendues : {expected_sources}")
    print(f"   Sources trouvées : {result['found']}")
    print(f"   Score : {result['score']:.0%}")

    assert result["score"] == 1.0, "Toutes les sources sont trouvées"
    assert len(result["found"]) == 2, "2 sources trouvées"
    assert len(result["missing"]) == 0, "Aucune source manquante"

    # Cas 2 : une source est manquante
    expected_sources = ["cours_rag.pdf", "cours_manquant.pdf"]
    result = evaluate_chunk_retrieval(retrieved_docs, expected_sources)

    print(f"\n   Sources attendues : {expected_sources}")
    print(f"   Sources trouvées : {result['found']}")
    print(f"   Sources manquantes : {result['missing']}")
    print(f"   Score : {result['score']:.0%}")

    assert result["score"] == 0.5, "50% des sources sont trouvées"
    assert len(result["found"]) == 1, "1 source trouvée"
    assert len(result["missing"]) == 1, "1 source manquante"

    print("   ✅ Rappel des chunks correct")
    print("   ✅ Test 6 réussi !\n")


# ---------------------------------------------------------------------------
# Test 7 : Chargement des questions
# ---------------------------------------------------------------------------


def test_load_questions():
    """
    Vérifie que le fichier de questions est correctement chargé.
    """
    print("=" * 60)
    print("🧪 Test 7 : Chargement des questions")
    print("=" * 60)

    data = load_questions()

    # Vérifications
    assert "questions" in data, "Le JSON doit contenir 'questions'"
    assert len(data["questions"]) == 15, f"15 questions attendues, {len(data['questions'])} trouvées"

    # Vérifier la structure d'une question
    q = data["questions"][0]
    assert "id" in q, "Chaque question doit avoir un id"
    assert "question" in q, "Chaque question doit avoir un texte"
    assert "expected_answer" in q, "Chaque question doit avoir une réponse attendue"
    assert "expected_sources" in q, "Chaque question doit avoir des sources attendues"
    assert "keywords" in q, "Chaque question doit avoir des mots-clés"
    assert "should_answer" in q, "Chaque question doit avoir should_answer"

    # Compter les types
    types = {}
    should_answer_count = 0
    for question in data["questions"]:
        t = question.get("type", "inconnu")
        types[t] = types.get(t, 0) + 1
        if question.get("should_answer", True):
            should_answer_count += 1

    print(f"   Nombre de questions : {len(data['questions'])}")
    print(f"   Types : {types}")
    print(f"   Questions avec réponse : {should_answer_count}")
    print(f"   Questions sans réponse : {len(data['questions']) - should_answer_count}")

    print("   ✅ Questions chargées correctement")
    print("   ✅ Test 7 réussi !\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("\n🧪 Tests de l'Étape 5 — Évaluation\n")
    print("   (Tests locaux, pas de clé API requise)\n")

    try:
        test_evaluate_citations()
        test_evaluate_invalid_citations()
        test_evaluate_relevance()
        test_evaluate_partial_relevance()
        test_evaluate_refusal()
        test_evaluate_chunk_retrieval()
        test_load_questions()

        print("=" * 60)
        print("🎉 Tous les tests de l'Étape 5 sont réussis !")
        print("=" * 60)
        print()
        print("📋 Pour lancer l'évaluation complète :")
        print("   1. Assure-toi d'avoir des PDF indexés")
        print("   2. python -m eval.evaluate")
        print("   3. Le rapport sera sauvegardé dans eval/rapport_evaluation.txt")
        print()

    except AssertionError as e:
        print(f"   ❌ Échec : {e}")
        exit(1)
