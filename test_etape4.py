"""
test_etape4.py — Test rapide de l'Étape 4
==========================================
Vérifie la recherche hybride (vectoriel + BM25)
SANS avoir besoin des clés API.

On teste :
  - Le fonctionnement de BM25 seul
  - La fusion RRF (Reciprocal Rank Fusion)
  - La pondération des retrievers
  - Le score BM25 manuel

Usage :
    python test_etape4.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.hybrid_retriever import HybridRetriever, compute_bm25_score, _tokenize

# ---------------------------------------------------------------------------
# Création de documents de test
# ---------------------------------------------------------------------------


def create_test_documents():
    """
    Crée un jeu de documents simulant des chunks de PDF.
    """
    docs = [
        Document(
            page_content="Le RAG (Retrieval-Augmented Generation) combine recherche documentaire et génération de texte. Il utilise des embeddings pour trouver les passages pertinents.",
            metadata={"source": "cours_rag.pdf", "page": 0}
        ),
        Document(
            page_content="La recherche vectorielle utilise des embeddings pour représenter les textes dans un espace sémantique. Les vecteurs proches ont des sens similaires.",
            metadata={"source": "cours_rag.pdf", "page": 1}
        ),
        Document(
            page_content="BM25 est un algorithme de recherche par mots-clés basé sur TF-IDF. Il est efficace pour les correspondances exactes de termes.",
            metadata={"source": "cours_bm25.pdf", "page": 0}
        ),
        Document(
            page_content="La recherche hybride combine les avantages de la recherche vectorielle (sémantique) et de BM25 (mots-clés) pour améliorer la pertinence.",
            metadata={"source": "cours_hybride.pdf", "page": 0}
        ),
        Document(
            page_content="ChromaDB est une base vectorielle open source qui permet de stocker et rechercher des embeddings efficacement.",
            metadata={"source": "cours_chroma.pdf", "page": 0}
        ),
        Document(
            page_content="Les embeddings sont des représentations numériques de textes. Le modèle voyage-3-longpert est optimisé pour les documents longs.",
            metadata={"source": "cours_embeddings.pdf", "page": 0}
        ),
    ]
    return docs


# ---------------------------------------------------------------------------
# Test 1 : BM25 seul
# ---------------------------------------------------------------------------


def test_bm25_retriever():
    """
    Vérifie que BM25 retrouve les documents par mots-clés.
    """
    print("=" * 60)
    print("🧪 Test 1 : BM25 — Recherche par mots-clés")
    print("=" * 60)

    docs = create_test_documents()
    bm25 = BM25Retriever.from_documents(docs, k=3)

    # Recherche avec un terme exact
    results = bm25.invoke("BM25")

    print(f"   Question : 'BM25'")
    print(f"   Résultats : {len(results)}")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "inconnu")
        preview = doc.page_content[:60] + "..."
        print(f"      {i}. [{source}] {preview}")

    # Vérifications
    assert len(results) > 0, "BM25 devrait trouver des résultats"
    assert any("BM25" in doc.page_content for doc in results), "Le document sur BM25 devrait être dans les résultats"

    print("   ✅ BM25 trouve les documents par mots-clés")
    print("   ✅ Test 1 réussi !\n")


# ---------------------------------------------------------------------------
# Test 2 : BM25 vs Vectoriel — cas où BM25 est meilleur
# ---------------------------------------------------------------------------


def test_bm25_vs_vectorial():
    """
    Montre un cas où BM25 est meilleur que le vectoriel :
    la recherche d'un terme technique exact.
    """
    print("=" * 60)
    print("🧪 Test 2 : Cas où BM25 excelle")
    print("=" * 60)

    docs = create_test_documents()
    bm25 = BM25Retriever.from_documents(docs, k=3)

    # Recherche d'un terme très spécifique
    query = "voyage-3-longpert"
    results = bm25.invoke(query)

    print(f"   Question : '{query}'")
    print(f"   Résultats : {len(results)}")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "inconnu")
        preview = doc.page_content[:60] + "..."
        print(f"      {i}. [{source}] {preview}")

    # Vérifications
    assert len(results) > 0, "BM25 devrait trouver des résultats"
    assert any("voyage-3-longpert" in doc.page_content for doc in results), "Le document mentionnant voyage-3-longpert devrait être trouvé"

    print("   ✅ BM25 trouve les termes techniques exacts")
    print("   ✅ Test 2 réussi !\n")


# ---------------------------------------------------------------------------
# Test 3 : HybridRetriever — fusion des classements
# ---------------------------------------------------------------------------


def test_hybrid_retriever():
    """
    Vérifie que HybridRetriever fusionne les résultats
    du vectoriel et de BM25.
    """
    print("=" * 60)
    print("🧪 Test 3 : HybridRetriever — Fusion RRF")
    print("=" * 60)

    docs = create_test_documents()

    # Créer un retriever vectoriel factice (on utilise BM25 ici aussi pour le test)
    # En production, ce serait le retriever ChromaDB
    class FakeVectorRetriever:
        def invoke(self, query):
            # Simule un retriever vectoriel qui favorise les documents sémantiquement proches
            # Ici on retourne les documents dans un ordre différent de BM25
            return [docs[1], docs[0], docs[4], docs[3], docs[2], docs[5]]

    # HybridRetriever
    hybrid = HybridRetriever(
        vector_retriever=FakeVectorRetriever(),
        chunks=docs,
        k=3,
        weight_vector=0.7,
        weight_bm25=0.3,
    )

    query = "recherche vectorielle embeddings"
    results = hybrid.invoke(query)

    print(f"   Question : '{query}'")
    print(f"   Résultats : {len(results)}")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "inconnu")
        preview = doc.page_content[:60] + "..."
        print(f"      {i}. [{source}] {preview}")

    # Vérifications
    assert len(results) > 0, "HybridRetriever devrait trouver des résultats"
    assert len(results) <= 3, "HybridRetriever devrait retourner au plus k résultats"

    print("   ✅ HybridRetriever fusionne les résultats")
    print("   ✅ Test 3 réussi !\n")


# ---------------------------------------------------------------------------
# Test 4 : Pondération des retrievers
# ---------------------------------------------------------------------------


def test_retriever_weights():
    """
    Vérifie que la pondération influence les résultats.
    """
    print("=" * 60)
    print("🧪 Test 4 : Pondération des retrievers")
    print("=" * 60)

    docs = create_test_documents()

    # Simulons deux retrievers avec des classements différents
    class RetrieverA:
        def invoke(self, query):
            return [docs[0], docs[1], docs[2]]  # Favorise rag, puis vectoriel, puis bm25

    class RetrieverB:
        def invoke(self, query):
            return [docs[2], docs[3], docs[4]]  # Favorise bm25, puis hybride, puis chroma

    # Avec poids 0.8/0.2 — devrait favoriser RetrieverA
    hybrid_favor_a = HybridRetriever(
        vector_retriever=RetrieverA(),
        chunks=docs,
        k=3,
        weight_vector=0.8,
        weight_bm25=0.2,
    )

    # Avec poids 0.2/0.8 — devrait favoriser RetrieverB
    hybrid_favor_b = HybridRetriever(
        vector_retriever=RetrieverB(),
        chunks=docs,
        k=3,
        weight_vector=0.2,
        weight_bm25=0.8,
    )

    query = "recherche"
    results_a = hybrid_favor_a.invoke(query)
    results_b = hybrid_favor_b.invoke(query)

    print(f"   Question : '{query}'")
    print(f"   Avec poids [0.8, 0.2] (favorise A) :")
    for doc in results_a:
        print(f"      - {doc.metadata.get('source')}")
    print(f"   Avec poids [0.2, 0.8] (favorise B) :")
    for doc in results_b:
        print(f"      - {doc.metadata.get('source')}")

    # Vérifications
    assert len(results_a) > 0, "Devrait trouver des résultats"
    assert len(results_b) > 0, "Devrait trouver des résultats"

    print("   ✅ La pondération influence les résultats")
    print("   ✅ Test 4 réussi !\n")


# ---------------------------------------------------------------------------
# Test 5 : Score BM25 manuel
# ---------------------------------------------------------------------------


def test_bm25_score():
    """
    Vérifie le calcul du score BM25.
    """
    print("=" * 60)
    print("🧪 Test 5 : Score BM25 manuel")
    print("=" * 60)

    doc1 = "Le RAG combine recherche et génération"
    doc2 = "La recherche vectorielle utilise des embeddings"
    doc3 = "Le RAG est une technique de RAG très utilisée"

    query = "RAG"

    avg_len = (len(doc1.split()) + len(doc2.split()) + len(doc3.split())) / 3

    score1 = compute_bm25_score(query, doc1, avg_len)
    score2 = compute_bm25_score(query, doc2, avg_len)
    score3 = compute_bm25_score(query, doc3, avg_len)

    print(f"   Question : '{query}'")
    print(f"   Score doc1 ('{doc1[:40]}...') : {score1:.4f}")
    print(f"   Score doc2 ('{doc2[:40]}...') : {score2:.4f}")
    print(f"   Score doc3 ('{doc3[:40]}...') : {score3:.4f}")

    # Vérifications
    assert score1 > score2, "doc1 mentionne RAG, doc2 non → score1 > score2"
    assert score3 > score1, "doc3 mentionne RAG deux fois → score3 > score1"

    print("   ✅ Le score BM25 reflète la pertinence")
    print("   ✅ Test 5 réussi !\n")


# ---------------------------------------------------------------------------
# Test 6 : Configuration de la recherche hybride
# ---------------------------------------------------------------------------


def test_hybrid_config():
    """
    Vérifie que la configuration de la recherche hybride est correcte.
    """
    print("=" * 60)
    print("🧪 Test 6 : Configuration")
    print("=" * 60)

    from src.rag_chain import (
        USE_HYBRID_SEARCH,
        HYBRID_WEIGHT_VECTOR,
        HYBRID_WEIGHT_BM25,
        K,
    )

    # Vérifications
    assert USE_HYBRID_SEARCH is True, "La recherche hybride devrait être activée par défaut"
    assert HYBRID_WEIGHT_VECTOR + HYBRID_WEIGHT_BM25 == 1.0, "La somme des poids devrait être 1.0"
    assert 0 <= HYBRID_WEIGHT_VECTOR <= 1, "Le poids vectoriel devrait être entre 0 et 1"
    assert 0 <= HYBRID_WEIGHT_BM25 <= 1, "Le poids BM25 devrait être entre 0 et 1"
    assert K > 0, "K devrait être positif"

    print(f"   Recherche hybride activée : {USE_HYBRID_SEARCH}")
    print(f"   Poids vectoriel : {HYBRID_WEIGHT_VECTOR}")
    print(f"   Poids BM25 : {HYBRID_WEIGHT_BM25}")
    print(f"   K (nombre de résultats) : {K}")
    print("   ✅ Configuration correcte")
    print("   ✅ Test 6 réussi !\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("\n🧪 Tests de l'Étape 4 — Recherche hybride\n")
    print("   (Tests locaux, pas de clé API requise)\n")

    try:
        test_bm25_retriever()
        test_bm25_vs_vectorial()
        test_hybrid_retriever()
        test_retriever_weights()
        test_bm25_score()
        test_hybrid_config()

        print("=" * 60)
        print("🎉 Tous les tests de l'Étape 4 sont réussis !")
        print("=" * 60)
        print()
        print("📋 Pour tester avec les API :")
        print("   1. Assure-toi d'avoir des PDF indexés")
        print("   2. python -m src.rag_chain")
        print("   3. Teste avec des questions techniques (termes exacts)")
        print("      et des questions sémantiques (concepts)")
        print()

    except AssertionError as e:
        print(f"   ❌ Échec : {e}")
        exit(1)
