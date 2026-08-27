import math
import re
from collections import Counter

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class HybridRetriever:

    def __init__(
        self,
        vector_retriever,
        chunks: list,
        k: int = 5,
        weight_vector: float = 0.7,
        weight_bm25: float = 0.3,
        bm25_k: int = 60,  # Paramètre de saturation BM25
    ):
        """
        Initialise le retriever hybride.

        Args:
            vector_retriever : retriever vectoriel (ChromaDB)
            chunks : liste des documents découpés
            k : nombre de résultats à retourner
            weight_vector : poids du vectoriel dans la fusion
            weight_bm25 : poids du BM25 dans la fusion
            bm25_k : paramètre de saturation BM25 (contrôle la sensibilité à la fréquence)
        """
        self.vector_retriever = vector_retriever
        self.k = k
        self.weight_vector = weight_vector
        self.weight_bm25 = weight_bm25
        self.bm25_k = bm25_k

        # Créer le retriever BM25
        self.bm25_retriever = BM25Retriever.from_documents(
            chunks,
            k=k,
        )

    def invoke(self, query: str) -> list:
        """
        Recherche hybride : combine les résultats vectoriels et BM25.

        Args:
            query : la question de l'utilisateur

        Returns:
            Liste des k meilleurs documents, triés par score RRF
        """
        # 1. Obtenir les résultats de chaque méthode
        vector_results = self.vector_retriever.invoke(query)
        bm25_results = self.bm25_retriever.invoke(query)

        # 2. Calculer les scores RRF
        rrf_scores = self._compute_rrf(vector_results, bm25_results)

        # 3. Trier par score RRF et retourner les k meilleurs
        sorted_docs = sorted(rrf_scores, key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:self.k]]

    def _compute_rrf(self, vector_results: list, bm25_results: list) -> list:
        """
        Calcule les scores RRF (Reciprocal Rank Fusion).

        RRF pour un document d :
            score_RRF(d) = Σ (weight_i / (k + rang_i(d)))

        où :
            weight_i = poids du retriever i
            rang_i(d) = rang du document d dans les résultats du retriever i
            k = paramètre de lissage (évite les scores extrêmes)

        Un document en tête de plusieurs listes aura un score élevé.
        Un document présent dans une seule liste aura un score modéré.

        Retourne une liste de tuples (document, score).
        """
        # Dictionnaire pour stocker les scores par document
        # On utilise le contenu du document comme clé unique
        scores = {}
        doc_map = {}

        # Scores du retriever vectoriel
        for rank, doc in enumerate(vector_results):
            key = self._doc_key(doc)
            if key not in scores:
                scores[key] = 0
                doc_map[key] = doc
            # Formule RRF : weight / (k_const + rank)
            scores[key] += self.weight_vector / (self.bm25_k + rank + 1)

        # Scores du retriever BM25
        for rank, doc in enumerate(bm25_results):
            key = self._doc_key(doc)
            if key not in scores:
                scores[key] = 0
                doc_map[key] = doc
            scores[key] += self.weight_bm25 / (self.bm25_k + rank + 1)

        # Convertir en liste de tuples
        return [(doc_map[key], score) for key, score in scores.items()]

    def _doc_key(self, doc: Document) -> str:
        """
        Génère une clé unique pour un document.

        On utilise le contenu + source + page pour éviter
        les collisions entre documents différents.
        """
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", 0)
        return f"{source}_{page}_{doc.page_content[:50]}"


def compute_bm25_score(query: str, document: str, avg_doc_length: float, k1: float = 1.5, b: float = 0.75) -> float:
    """
    Calcule le score BM25 entre une question et un document.

    BM25 est basé sur TF-IDF avec une saturation de la fréquence
    et une normalisation par longueur.

    Formule :
        score = Σ IDF(t) * (f(t,D) * (k1 + 1)) / (f(t,D) + k1 * (1 - b + b * |D|/avgdl))

    où :
        f(t,D) = fréquence du terme t dans le document D
        |D| = longueur du document
        avgdl = longueur moyenne des documents
        k1 = paramètre de saturation (1.5 par défaut)
        b = paramètre de normalisation par longueur (0.75 par défaut)
    """
    # Tokeniser
    query_terms = _tokenize(query)
    doc_terms = _tokenize(document)

    if not doc_terms:
        return 0.0

    # Fréquences des termes dans le document
    term_freqs = Counter(doc_terms)
    doc_length = len(doc_terms)

    score = 0.0
    for term in query_terms:
        # IDF simplifié (sans collection de référence)
        # Ici on utilise une heuristique : les termes rares dans le document ont plus de poids
        idf = math.log(1 + (len(doc_terms) - term_freqs[term] + 0.5) / (term_freqs[term] + 0.5))

        # Fréquence normalisée
        tf = term_freqs[term]
        normalized_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))

        score += idf * normalized_tf

    return score


def _tokenize(text: str) -> list:
    """
    Tokenise un texte en minuscules, en retirant la ponctuation.
    """
    return re.findall(r'\b\w+\b', text.lower())
