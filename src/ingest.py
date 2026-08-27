
# ingest.py — Pipeline d'ingestion des PDF


import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

# Dossier contenant les PDF à indexer
DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"

# Dossier où ChromaDB persiste ses données (fichiers sur disque)
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Nom de la collection dans ChromaDB
COLLECTION_NAME = "documents"

# Modèle d'embeddings Voyage AI
# voyage-4 : generaliste/multilingue, 32k tokens de contexte, 1024 dimensions.
# ATTENTION : ce modele doit etre STRICTEMENT le meme que dans rag_chain.py.
# Indexer avec un modele et interroger avec un autre donne des resultats absurdes,
# car les deux ne placent pas les textes dans le meme espace vectoriel.
EMBEDDING_MODEL = "voyage-4"

# Paramètres de découpage
CHUNK_SIZE = 1000       # Taille cible de chaque chunk (en caractères)
CHUNK_OVERLAP = 200     # Chevauchement entre chunks consécutifs

# ---------------------------------------------------------------------------
# Limites de debit (rate limits) Voyage AI
# ---------------------------------------------------------------------------

BATCH_SIZE = 30

# Pause entre deux lots. A 3 RPM il faut 20 s entre requetes ; on prend 25 s
# de marge pour absorber la latence reseau.
DELAI_ENTRE_LOTS = 25

# Nombre de tentatives par lot avant abandon.
MAX_TENTATIVES = 5

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def load_pdfs(directory: Path) -> list:
    """
    Charge tous les fichiers PDF d'un dossier.

    PyPDFLoader crée un Document par page, avec :
      - page_content : le texte de la page
      - metadata : {"source": "fichier.pdf", "page": 0}

    Retourne une liste de Documents (un par page de tous les PDF).
    """
    all_docs = []
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        print(f"  Aucun PDF trouvé dans {directory}")
        print("   Dépose des fichiers PDF dans le dossier documents/ et relance.")
        return []

    print(f" {len(pdf_files)} PDF trouvé(s) :")
    for pdf_path in pdf_files:
        print(f"   - {pdf_path.name}")

    for pdf_path in tqdm(pdf_files, desc="Chargement des PDF"):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)

    print(f"   → {len(all_docs)} pages chargées au total.")
    return all_docs


def split_documents(docs: list) -> list:
    """
    Découpe les documents en chunks plus petits.

    Le RecursiveCharacterTextSplitter essaie de couper aux limites
    naturelles du texte (paragraphes → lignes → mots) pour préserver
    le sens.

    Paramètres clés :
      - chunk_size : taille cible en caractères (pas en tokens !)
      - chunk_overlap : caractères partagés entre deux chunks voisins
        pour éviter de couper une phrase en deux.

    Les métadonnées (source, page) sont conservées sur chaque chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Séparateurs essayés dans l'ordre : paragraphes, lignes, mots
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    print(f"\n  Découpage : {len(docs)} pages → {len(chunks)} chunks")
    print(f"   chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    return chunks


def index_in_chroma(chunks: list) -> Chroma:
    """
    Génère les embeddings des chunks et les stocke dans ChromaDB, PAR LOTS.

    VoyageAIEmbeddings transforme chaque chunk en un vecteur de 1024
    dimensions. ChromaDB indexe ces vecteurs pour permettre une recherche
    de similarité rapide.

    Pourquoi par lots plutot qu'en un seul appel :
    sans carte bancaire, Voyage limite le debit a 3 requetes/minute et
    10 000 tokens/minute. Un `Chroma.from_documents(chunks)` avec 142 chunks
    envoie tout d'un coup et se fait refuser (RateLimitError).

    Strategie :
      1. Creer la collection vide
      2. Ajouter les chunks par paquets de BATCH_SIZE
      3. Pauser DELAI_ENTRE_LOTS secondes entre chaque paquet
      4. En cas de RateLimitError, attendre de plus en plus longtemps
         (backoff exponentiel) et reessayer jusqu'a MAX_TENTATIVES

    persist_directory : les données sont sauvegardées sur disque,
    donc pas besoin de ré-indexer à chaque lancement.
    """
    print(f"\n Génération des embeddings avec {EMBEDDING_MODEL}")
    embeddings = VoyageAIEmbeddings(model=EMBEDDING_MODEL)

    # Créer la collection (vide au départ)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    lots = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    duree_estimee = (len(lots) - 1) * DELAI_ENTRE_LOTS / 60

    print(" Indexation par lots dans ChromaDB")
    print(f"   {len(chunks)} chunks → {len(lots)} lots de {BATCH_SIZE} max")
    print(f"   Pause de {DELAI_ENTRE_LOTS}s entre les lots (limite : 3 req/min)")
    print(f"   Durée estimée : ~{duree_estimee:.0f} min\n")

    total_indexes = 0

    for numero, lot in enumerate(lots, start=1):
        for tentative in range(1, MAX_TENTATIVES + 1):
            try:
                vectorstore.add_documents(lot)
                total_indexes += len(lot)
                print(
                    f"   [{numero:2}/{len(lots)}] {len(lot):3} chunks indexés "
                    f"→ total {total_indexes}/{len(chunks)}"
                )
                break  # lot réussi, on sort de la boucle de tentatives

            except Exception as exc:
                message = str(exc)
                est_rate_limit = (
                    "rate limit" in message.lower()
                    or "429" in message
                    or "RateLimit" in type(exc).__name__
                )

                if not est_rate_limit:
                    # Erreur non liée au débit : inutile de réessayer
                    print(f"    Lot {numero} : {type(exc).__name__}")
                    print(f"      {message[:200]}")
                    raise

                if tentative == MAX_TENTATIVES:
                    print(
                        f"    Lot {numero} abandonné après "
                        f"{MAX_TENTATIVES} tentatives."
                    )
                    print(f"      {total_indexes} chunks indexés avant l'échec.")
                    print("      Supprime chroma_db/ et relance pour repartir à zéro.")
                    raise

                # Backoff exponentiel : 25s, 50s, 100s, 200s...
                attente = DELAI_ENTRE_LOTS * (2 ** (tentative - 1))
                print(
                    f"    Lot {numero} : limite de débit atteinte "
                    f"(tentative {tentative}/{MAX_TENTATIVES}), "
                    f"attente {attente}s..."
                )
                time.sleep(attente)

        # Pause avant le lot suivant (inutile après le dernier)
        if numero < len(lots):
            time.sleep(DELAI_ENTRE_LOTS)

    print(f"\n   → {total_indexes} chunks indexés dans {CHROMA_DIR}")
    return vectorstore


def main():
    """
    Pipeline complet : charger → découper → indexer.
    """
    # Charger les variables d'environnement (.env)
    load_dotenv()

    # Vérifier que les clés API sont présentes
    if not os.getenv("VOYAGE_API_KEY"):
        print(" VOYAGE_API_KEY manquante. Vérifie ton fichier .env")
        return

    print("=" * 60)
    print(" Pipeline d'ingestion RAG")
    print("=" * 60)

    # Étape 1 : charger les PDF
    docs = load_pdfs(DOCUMENTS_DIR)
    if not docs:
        return

    # Étape 2 : découper en chunks
    chunks = split_documents(docs)

    # Étape 3 : indexer dans ChromaDB
    index_in_chroma(chunks)

    print("\n Ingestion terminée ! Tu peux maintenant lancer rag_chain.py")


if __name__ == "__main__":
    main()
