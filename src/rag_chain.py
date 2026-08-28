
# rag_chain.py — Chaîne RAG avec citations, reformulation et recherche hybride

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.hybrid_retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "documents"

# Modèle d'embeddings Voyage AI
# voyage-4 : generaliste/multilingue, 32k de contexte, 1024 dimensions par defaut.
# Alternatives : voyage-4-large (meilleure qualite), voyage-4-lite (moins cher),
# voyage-law-2 / voyage-finance-2 (specialises par domaine).
# IMPORTANT : ne jamais changer ce modele sans re-indexer tout Chroma. Les vecteurs
# d'un modele ne sont pas comparables a ceux d'un autre.
EMBEDDING_MODEL = "voyage-4"

# Paramètre du retriever : nombre de chunks récupérés
K = 5

# Modèle LLM principal (redaction de la reponse citee)
# gemini-3.6-flash : version STABLE, palier gratuit permanent.
# Preferer un id "stable" a un id "-preview" : les previews sont deprecies
# avec seulement 2 semaines de preavis.
LLM_MODEL = "gemini-3.6-flash"

SUPPORTS_TEMPERATURE = False
LLM_TEMPERATURE = 0

# Modèle pour la reformulation (plus rapide et moins cher)
# gemini-3.5-flash-lite : le plus rapide de la famille, largement suffisant
# pour reecrire une question. Utiliser un modele leger ici economise du quota
# gratuit pour le modele principal.
REFORMULATOR_MODEL = "gemini-3.5-flash-lite"

# Activer/désactiver la reformulation contextuelle
USE_REFORMULATION = True

# Activer/désactiver la recherche hybride (Étape 4)
USE_HYBRID_SEARCH = True

HYBRID_WEIGHT_VECTOR = 0.7
HYBRID_WEIGHT_BM25 = 0.3

SYSTEM_PROMPT = """Tu es un assistant documentaire factuel.
Tu réponds exclusivement à partir des documents fournis dans le contexte ci-dessous.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT si l'information est présente dans le contexte.
   Si la réponse n'est pas dans le contexte, dis clairement : "Je ne trouve pas cette information dans les documents fournis."
2. Cite tes sources après chaque affirmation avec le format : [nom_du_fichier, page N]
3. Ne jamais inventer, supposer ou extrapoler au-delà du contexte.
4. Si le contexte contient des informations contradictoires, le signaler.
5. Réponds dans la même langue que la question.

CONTEXTE :
{context}"""

# Prompt de reformulation contextuelle (Étape 3)
# Ce prompt est utilisé par le "reformulateur" — un LLM rapide qui
# transforme une question dépendante du contexte en question autonome.
#
# Exemple :
#   Entrée : "Et pour le deuxième cas ?"
#   Sortie : "Qu'est-ce que l'IA forte et quelles sont ses caractéristiques ?"
#
# C'est nécessaire car la recherche vectorielle (retriever) a besoin
# d'une question complète et autonome pour trouver les bons chunks.

REFORMULATION_PROMPT = """Tu es un assistant qui reformule les questions pour les rendre autonomes.

RÈGLE :
- Si la question est déjà autonome (elle contient tous les éléments nécessaires), retourne-la telle quelle.
- Si la question dépend du contexte (pronoms, références implicites, "cela", "ce dernier", "le deuxième cas", etc.), reformule-la en une question complète et autonome en utilisant l'historique de conversation.
- Ne réponds PAS à la question. Reformule-la uniquement.
- Retourne UNIQUEMENT la question reformulée, sans explication ni ponctuation supplémentaire.

HISTORIQUE DE CONVERSATION :
{history}

QUESTION À REFORMULER :
{question}

QUESTION REFORMULÉE :"""


# Fonctions

def format_docs(docs: list) -> str:
    """
    Formate les chunks récupérés en texte injectable dans le prompt.

    Chaque chunk est présenté avec sa source et sa page,
    pour que le LLM puisse citer correctement.

    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "inconnu")
        # Le source contient le chemin complet, on garde juste le nom
        source_name = os.path.basename(source)
        page = doc.metadata.get("page", 0) + 1  # +1 car pages indexées à 0
        formatted.append(f"[{source_name}, page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def load_vectorstore() -> Chroma:
    """
    Charge la base vectorielle ChromaDB existante.

    On utilise le même modèle d'embeddings que pour l'ingestion,
    sinon les vecteurs ne sont pas dans le même espace.
    """
    embeddings = VoyageAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Vérifier que la collection n'est pas vide
    count = vectorstore._collection.count()
    if count == 0:
        print("La base vectorielle est vide.")
        print("   Lance d'abord : python -m src.ingest")
        return None

    print(f"Base vectorielle chargée : {count} chunks indexés")
    return vectorstore


def build_hybrid_retriever(vectorstore: Chroma, chunks: list) -> HybridRetriever:
    """
    Construit un retriever hybride combinant recherche vectorielle et BM25.

    Args:
        vectorstore: la base vectorielle ChromaDB
        chunks: la liste des documents découpés (pour BM25)

    Retourne un HybridRetriever qui combine les deux méthodes.
    """
    # Retriever vectoriel
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": K},
    )

    # HybridRetriever avec fusion RRF
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        chunks=chunks,
        k=K,
        weight_vector=HYBRID_WEIGHT_VECTOR,
        weight_bm25=HYBRID_WEIGHT_BM25,
    )

    print(f"Recherche hybride activée : {HYBRID_WEIGHT_VECTOR*100:.0f}% vectoriel + {HYBRID_WEIGHT_BM25*100:.0f}% BM25")
    return hybrid_retriever


def build_rag_chain(vectorstore: Chroma, chunks: list = None, use_reformulation: bool = True):
    """
    Construit la chaîne RAG avec LCEL (LangChain Expression Language).

    La chaîne est composée de 5 maillons (avec reformulation) :

    1. reformulateur (optionnel) : transforme une question dépendante
       du contexte en question autonome, en s'appuyant sur l'historique.

    2. retriever : prend la question (reformulée), calcule son embedding,
       retrouve les K chunks les plus similaires dans ChromaDB.
       Si USE_HYBRID_SEARCH est activé, combine vectoriel + BM25.

    3. format_docs : transforme la liste de Documents en texte brut
       avec les métadonnées de citation.

    4. prompt : injecte le contexte formaté dans le template système.
    5. LLM : génère la réponse en suivant les instructions du prompt.

    L'opérateur | (pipe) passe la sortie d'un maillon à l'entrée du suivant.
    """
    # --- Retriever ---
    # Si la recherche hybride est activée et les chunks sont disponibles,
    # on utilise EnsembleRetriever (vectoriel + BM25)
    if USE_HYBRID_SEARCH and chunks is not None:
        retriever = build_hybrid_retriever(vectorstore, chunks)
    else:
        # Recherche vectorielle seule
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": K},
        )
        if USE_HYBRID_SEARCH and chunks is None:
            print("Recherche hybride activée mais chunks non fournis. Utilisation du vectoriel seul.")
            print("   Pour activer la recherche hybride, passe les chunks à build_rag_chain.")

    # LLM principal 
    # On ne passe temperature que si le modele l'accepte (voir
    # SUPPORTS_TEMPERATURE en tete de fichier). Ce petit dictionnaire evite
    # d'avoir deux branches d'instanciation quasi identiques.
    options_llm = {"temperature": LLM_TEMPERATURE} if SUPPORTS_TEMPERATURE else {}

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        **options_llm,
    )

    # --- Prompt principal ---
    # ChatPromptTemplate permet de créer des templates avec rôles
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # --- Reformulateur (Étape 3) ---
    # Un LLM plus rapide qui transforme les questions dépendantes
    # du contexte en questions autonomes.
    reformulator = ChatGoogleGenerativeAI(
        model=REFORMULATOR_MODEL,
        **options_llm,
    )
    reformulation_prompt = ChatPromptTemplate.from_template(REFORMULATION_PROMPT)

    # --- Assemblage de la chaîne (LCEL) ---
    # Le dictionnaire en entrée est décomposé :
    #   - "context" ← retriever → format_docs
    #   - "question" ← RunnablePassthrough (tel quel)
    # Puis le prompt et le LLM prennent le relais.

    if use_reformulation:
        # Avec reformulation : on ajoute le maillon avant le retriever
        # Le reformulateur prend la question + l'historique et produit
        # une question autonome, qui est ensuite passée au retriever.
        rag_chain = (
            {
                "context": (
                    {"question": RunnablePassthrough(), "history": RunnablePassthrough()}
                    | reformulation_prompt
                    | reformulator
                    | StrOutputParser()
                    | retriever
                    | format_docs
                ),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        # Sans reformulation (comportement Étape 1)
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    return rag_chain


def ask_question(chain, question: str, history: str = "") -> str:
    """
    Pose une question à la chaîne RAG et retourne la réponse.

    Si la reformulation est activée, l'historique est passé
    au reformulateur pour contextualiser la question.
    """
    print(f"\n Question : {question}")
    print("-" * 60)

    # invoke() exécute toute la chaîne
    response = chain.invoke(question)

    print(f" Réponse :\n{response}")
    return response


def main():
    """
    Boucle interactive de questions-réponses.
    """
    load_dotenv()

    # Vérifier les clés API
    if not os.getenv("GOOGLE_API_KEY"):
        print(" GOOGLE_API_KEY manquante. Vérifie ton fichier .env")
        return

    print("=" * 60)
    print(" Assistant RAG — Mode interactif")
    print(" Tape 'quit' pour sortir")
    print(" Reformulation contextuelle : activée" if USE_REFORMULATION else "   Reformulation contextuelle : désactivée")
    print(" Recherche hybride : activée" if USE_HYBRID_SEARCH else "   Recherche hybride : désactivée")
    print("=" * 60)

    # Charger la base vectorielle
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return

    # Charger les chunks pour BM25 (si recherche hybride activée)
    chunks = None
    if USE_HYBRID_SEARCH:
        print("Chargement des chunks pour BM25...")
        from src.ingest import load_pdfs, split_documents
        documents_dir = Path(__file__).parent.parent / "documents"
        docs = load_pdfs(documents_dir)
        if docs:
            chunks = split_documents(docs)
            print(f"   → {len(chunks)} chunks chargés pour BM25")
        else:
            print("Aucun PDF trouvé. Recherche vectorielle seule.")

    # Construire la chaîne RAG
    rag_chain = build_rag_chain(vectorstore, chunks=chunks, use_reformulation=USE_REFORMULATION)

    # Historique local pour la reformulation
    history = []

    # Boucle de questions
    while True:
        question = input("\nTa question > ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        if not question:
            continue

        # Construire l'historique textuel
        history_text = ""
        if history:
            lines = []
            for h in history:
                lines.append(f"Utilisateur: {h['question']}")
                lines.append(f"Assistant: {h['answer']}")
            history_text = "\n".join(lines)

        # Appeler la chaîne avec ou sans reformulation
        if USE_REFORMULATION:
            response = rag_chain.invoke({"question": question, "history": history_text})
        else:
            response = rag_chain.invoke(question)

        print(f"\n Réponse :\n{response}")

        # Sauvegarder dans l'historique
        history.append({"question": question, "answer": response})


if __name__ == "__main__":
    main()
