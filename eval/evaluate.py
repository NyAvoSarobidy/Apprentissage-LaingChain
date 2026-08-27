
# evaluate.py — Script d'évaluation du système RAG

import json
import os
import re
import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.hybrid_retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "voyage-4"
K = 5
LLM_MODEL = "gemini-3.6-flash"
LLM_TEMPERATURE = 0
REFORMULATOR_MODEL = "gemini-3.5-flash-lite"

# Poids pour le score global
WEIGHT_CITATIONS = 0.25
WEIGHT_RECALL = 0.25
WEIGHT_REFUSAL = 0.25
WEIGHT_RELEVANCE = 0.25

# ---------------------------------------------------------------------------
# Fonctions d'évaluation
# ---------------------------------------------------------------------------


def load_questions(path: str = "eval/questions.json") -> dict:
    """Charge le jeu de questions-références."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vectorstore() -> Chroma:
    """Charge la base vectorielle."""
    embeddings = VoyageAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def build_rag_chain(vectorstore: Chroma, chunks: list):
    """Construit la chaîne RAG avec reformulation et recherche hybride."""
    # Retriever vectoriel
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": K},
    )

    # Hybrid retriever
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        chunks=chunks,
        k=K,
        weight_vector=0.7,
        weight_bm25=0.3,
    )

    # LLM principal
    # temperature non passe : les modeles Gemini 3.x ont un sampling fixe
    # (voir SUPPORTS_TEMPERATURE dans src/rag_chain.py).
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

    # Prompt système
    system_prompt = """Tu es un assistant documentaire factuel.
Tu réponds exclusivement à partir des documents fournis dans le contexte ci-dessous.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT si l'information est présente dans le contexte.
   Si la réponse n'est pas dans le contexte, dis clairement : "Je ne trouve pas cette information dans les documents fournis."
2. Cite tes sources après chaque affirmation avec le format : [nom_du_fichier, page N]
3. Ne jamais inventer, supposer ou extrapoler au-delà du contexte.
4. Réponds dans la même langue que la question.

CONTEXTE :
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # Format docs
    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "inconnu")
            source_name = os.path.basename(source)
            page = doc.metadata.get("page", 0) + 1
            formatted.append(f"[{source_name}, page {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    # Chaîne RAG
    rag_chain = (
        {
            "context": hybrid_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def evaluate_citations(response: str, expected_sources: list) -> dict:
    """
    Évalue la précision des citations.

    Vérifie que les sources citées dans la réponse existent
    dans les sources attendues.

    Retourne : {"score": float, "cited": list, "valid": list, "invalid": list}
    """
    # Extraire les citations du format [source, page N]
    citation_pattern = r'\[([^\]]+),\s*page\s*(\d+)\]'
    citations = re.findall(citation_pattern, response)

    if not citations:
        return {"score": 0.0, "cited": [], "valid": [], "invalid": []}

    cited_sources = [c[0] for c in citations]
    valid = [s for s in cited_sources if any(exp in s for exp in expected_sources)]
    invalid = [s for s in cited_sources if not any(exp in s for exp in expected_sources)]

    score = len(valid) / len(cited_sources) if cited_sources else 0.0

    return {
        "score": score,
        "cited": cited_sources,
        "valid": valid,
        "invalid": invalid,
    }


def evaluate_chunk_retrieval(retrieved_docs: list, expected_sources: list) -> dict:
    """
    Évalue le rappel des chunks.

    Vérifie que les chunks récupérés incluent les sources attendues.

    Retourne : {"score": float, "found": list, "missing": list}
    """
    if not expected_sources:
        return {"score": 1.0, "found": [], "missing": []}

    found = []
    missing = []

    for source in expected_sources:
        if any(source in doc.metadata.get("source", "") for doc in retrieved_docs):
            found.append(source)
        else:
            missing.append(source)

    score = len(found) / len(expected_sources) if expected_sources else 1.0

    return {"score": score, "found": found, "missing": missing}


def evaluate_relevance(response: str, keywords: list) -> dict:
    """
    Évalue la pertinence de la réponse.

    Compte les mots-clés attendus présents dans la réponse.

    Retourne : {"score": float, "found": list, "missing": list}
    """
    if not keywords:
        return {"score": 1.0, "found": [], "missing": []}

    response_lower = response.lower()
    found = [kw for kw in keywords if kw.lower() in response_lower]
    missing = [kw for kw in keywords if kw.lower() not in response_lower]

    score = len(found) / len(keywords) if keywords else 1.0

    return {"score": score, "found": found, "missing": missing}


def evaluate_refusal(response: str, should_answer: bool) -> dict:
    """
    Évalue si le système refuse correctement quand il faut.

    Retourne : {"correct": bool, "score": float}
    """
    refusal_phrases = [
        "je ne trouve pas",
        "je ne sais pas",
        "pas dans les documents",
        "pas dans le contexte",
        "information n'est pas disponible",
    ]

    has_refusal = any(phrase in response.lower() for phrase in refusal_phrases)

    if should_answer:
        # Le système devrait répondre, pas refuser
        correct = not has_refusal
    else:
        # Le système devrait refuser, pas répondre
        correct = has_refusal

    return {"correct": correct, "score": 1.0 if correct else 0.0}


def evaluate_single_question(chain, retriever: HybridRetriever, question: dict) -> dict:
    """
    Évalue une seule question.

    Retourne un dict avec toutes les métriques.
    """
    q_text = question["question"]
    expected_sources = question.get("expected_sources", [])
    keywords = question.get("keywords", [])
    should_answer = question.get("should_answer", True)

    # Récupérer les chunks (pour évaluation du rappel)
    retrieved_docs = retriever.invoke(q_text)

    # Obtenir la réponse
    response = chain.invoke(q_text)

    # Évaluer chaque métrique
    citation_result = evaluate_citations(response, expected_sources)
    retrieval_result = evaluate_chunk_retrieval(retrieved_docs, expected_sources)
    relevance_result = evaluate_relevance(response, keywords)
    refusal_result = evaluate_refusal(response, should_answer)

    # Score global
    global_score = (
        WEIGHT_CITATIONS * citation_result["score"]
        + WEIGHT_RECALL * retrieval_result["score"]
        + WEIGHT_REFUSAL * refusal_result["score"]
        + WEIGHT_RELEVANCE * relevance_result["score"]
    )

    return {
        "question_id": question["id"],
        "question": q_text,
        "response": response,
        "global_score": global_score,
        "citations": citation_result,
        "retrieval": retrieval_result,
        "relevance": relevance_result,
        "refusal": refusal_result,
    }


def print_report(results: list, output_path: str = None):
    """
    Affiche et sauvegarde le rapport d'évaluation.
    """
    report_lines = []

    def add(line=""):
        report_lines.append(line)
        print(line)

    add("=" * 70)
    add("RAPPORT D'ÉVALUATION DU SYSTÈME RAG")
    add("=" * 70)
    add()

    # Scores globaux
    global_scores = [r["global_score"] for r in results]
    avg_global = sum(global_scores) / len(global_scores) if global_scores else 0

    citation_scores = [r["citations"]["score"] for r in results]
    retrieval_scores = [r["retrieval"]["score"] for r in results]
    relevance_scores = [r["relevance"]["score"] for r in results]
    refusal_scores = [r["refusal"]["score"] for r in results]

    add("SCORES GLOBAUX")
    add("-" * 70)
    add(f"   Score global moyen        : {avg_global:.2%}")
    add(f"   Précision des citations   : {sum(citation_scores)/len(citation_scores):.2%}")
    add(f"   Rappel des chunks         : {sum(retrieval_scores)/len(retrieval_scores):.2%}")
    add(f"   Pertinence des réponses   : {sum(relevance_scores)/len(relevance_scores):.2%}")
    add(f"   Taux de refus correct     : {sum(refusal_scores)/len(refusal_scores):.2%}")
    add()

    # Détails par question
    add("DÉTAILS PAR QUESTION")
    add("-" * 70)

    for r in results:
        score_bar = "█" * int(r["global_score"] * 20) + "░" * (20 - int(r["global_score"] * 20))
        add(f"\n   Q{r['question_id']:02d} | {score_bar} | {r['global_score']:.0%}")
        add(f"   Q: {r['question']}")
        add(f"   R: {r['response'][:100]}...")

        # Citations
        if r["citations"]["cited"]:
            add(f"    Citations: {len(r['citations']['valid'])}/{len(r['citations']['cited'])} valides")

        # Retrieval
        if r["retrieval"]["found"]:
            add(f"    Chunks trouvés: {', '.join(r['retrieval']['found'])}")
        if r["retrieval"]["missing"]:
            add(f"    Chunks manquants: {', '.join(r['retrieval']['missing'])}")

        # Refusal
        if not r["refusal"]["correct"]:
            add(f"     Refusal incorrect")

    add()
    add("=" * 70)

    # Sauvegarder le rapport
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\n Rapport sauvegardé : {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Lance l'évaluation complète."""
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        print(" GOOGLE_API_KEY manquante. Vérifie ton fichier .env")
        return

    # Charger les questions
    questions_data = load_questions()
    questions = questions_data["questions"]
    print(f" {len(questions)} questions chargées")

    # Charger la base vectorielle
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return

    # Charger les chunks pour BM25
    from src.ingest import load_pdfs, split_documents
    documents_dir = Path(__file__).parent.parent / "documents"
    docs = load_pdfs(documents_dir)
    if not docs:
        print(" Aucun PDF trouvé dans documents/")
        return

    chunks = split_documents(docs)

    # Construire la chaîne RAG
    chain = build_rag_chain(vectorstore, chunks)

    # Construire le retriever pour l'évaluation
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": K},
    )
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        chunks=chunks,
        k=K,
        weight_vector=0.7,
        weight_bm25=0.3,
    )

    # Évaluer chaque question
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n Évaluation Q{i:02d}/{len(questions)}: {question['question'][:50]}...")
        result = evaluate_single_question(chain, retriever, question)
        results.append(result)
        print(f"   Score: {result['global_score']:.0%}")

    # Afficher le rapport
    output_path = Path(__file__).parent / "rapport_evaluation.txt"
    print_report(results, output_path=str(output_path))


if __name__ == "__main__":
    main()
