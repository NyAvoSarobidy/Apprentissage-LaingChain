
# test_etape1.py — Test rapide de l'Étape 1


from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Test 1 : Découpage d'un texte long en chunks
# ---------------------------------------------------------------------------


def test_split_documents():
    """
    Vérifie que le découpage produit des chunks de taille raisonnable
    et conserve les métadonnées.
    """
    print("=" * 60)
    print(" Test 1 : Découpage en chunks")
    print("=" * 60)

    # Simulons un document de 3 pages
    long_text = """
    L'intelligence artificielle (IA) est un ensemble de technologies qui permettent
    à des machines d'accomplir des tâches qui nécessitent normalement l'intelligence
    humaine. Ces tâches incluent la reconnaissance de la voix, la prise de décision,
    la traduction de langues et la reconnaissance visuelle.

    L'IA est divisée en deux catégories principales : l'IA faible et l'IA forte.
    L'IA faible, aussi appelée IA étroite, est conçue pour accomplir une tâche
    spécifique. Les assistants vocaux comme Siri et Alexa sont des exemples d'IA faible.
    L'IA forte, ou intelligence artificielle générale (IAG), serait capable de comprendre
    et d'apprendre n'importe quelle tâche intellectuelle qu'un humain peut accomplir.

    Les applications de l'IA sont nombreuses et variées. Dans le domaine de la santé,
    l'IA aide au diagnostic médical et à la découverte de médicaments. Dans le secteur
    financier, elle est utilisée pour la détection de fraudes et le trading algorithmique.
    Les véhicules autonomes utilisent l'IA pour percevoir leur environnement et naviguer
    en toute sécurité. Enfin, les systèmes de recommandation sur les plateformes de
    streaming utilisent l'IA pour personnaliser le contenu proposé aux utilisateurs.
    """ * 3  # Répéter pour avoir assez de texte

    doc = Document(
        page_content=long_text,
        metadata={"source": "test_ia.pdf", "page": 0}
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents([doc])

    print(f" Texte original : {len(long_text)} caractères")
    print(f" Chunks produits : {len(chunks)}")

    # Vérifications
    assert len(chunks) > 1, "Le découpage devrait produire plusieurs chunks"

    for i, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "test_ia.pdf", "Métadonnées source perdues"
        assert chunk.metadata["page"] == 0, "Métadonnées page perdues"
        assert len(chunk.page_content) <= 1200, f"Chunk {i} trop grand: {len(chunk.page_content)}"

    print(f" Tous les chunks ont les métadonnées correctes")
    print(f" Taille max d'un chunk : {max(len(c.page_content) for c in chunks)} caractères")
    print("  Test 1 réussi !\n")


# ---------------------------------------------------------------------------
# Test 2 : Formatage des citations
# ---------------------------------------------------------------------------


def test_format_docs():
    """
    Vérifie que le formatage des documents produit un texte
    avec les citations [source, page N] correctement formatées.
    """
    print("=" * 60)
    print("Test 2 : Formatage des citations")
    print("=" * 60)

    # Simulons des chunks récupérés
    docs = [
        Document(
            page_content="L'IA faible est conçue pour une tâche spécifique.",
            metadata={"source": "/chemin/vers/cours_ia.pdf", "page": 0}
        ),
        Document(
            page_content="L'IA forte serait capable de comprendre n'importe quelle tâche.",
            metadata={"source": "/chemin/vers/cours_ia.pdf", "page": 1}
        ),
        Document(
            page_content="Les véhicules autonomes utilisent l'IA pour naviguer.",
            metadata={"source": "/chemin/vers/rapport_auto.pdf", "page": 5}
        ),
    ]

    # Reproduire la logique de format_docs de rag_chain.py
    import os
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "inconnu")
        source_name = os.path.basename(source)
        page = doc.metadata.get("page", 0) + 1
        formatted.append(f"[{source_name}, page {page}]\n{doc.page_content}")

    result = "\n\n".join(formatted)

    print("   Résultat du formatage :")
    print("   " + "-" * 40)
    for line in result.split("\n"):
        print(f"   {line}")
    print("   " + "-" * 40)

    # Vérifications
    assert "[cours_ia.pdf, page 1]" in result, "Citation page 1 manquante"
    assert "[cours_ia.pdf, page 2]" in result, "Citation page 2 manquante"
    assert "[rapport_auto.pdf, page 6]" in result, "Citation page 6 manquante"
    assert "/chemin/vers/" not in result, "Le chemin complet ne devrait pas apparaître"

    print("   Les citations sont au bon format [source, page N]")
    print("   Seul le nom du fichier apparaît (pas le chemin complet)")
    print("   Test 2 réussi !\n")


# ---------------------------------------------------------------------------
# Test 3 : Vérification de l'overlap
# ---------------------------------------------------------------------------


def test_overlap():
    """
    Vérifie que les chunks consécutifs ont bien un chevauchement.
    """
    print("=" * 60)
    print("Test 3 : Chevauchement (overlap)")
    print("=" * 60)

    text = "Mot1 Mot2 Mot3 Mot4 Mot5 Mot6 Mot7 Mot8 Mot9 Mot10 " * 20
    doc = Document(page_content=text, metadata={"source": "test.pdf", "page": 0})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=30,
        separators=[" "],
    )

    chunks = splitter.split_documents([doc])

    print(f"   Chunks produits : {len(chunks)}")

    # Vérifier qu'il y a du contenu partagé entre chunks consécutifs
    has_overlap = False
    for i in range(len(chunks) - 1):
        # Prendre les 20 derniers caractères du chunk i
        fin_chunk_i = chunks[i].page_content[-20:]
        # Vérifier s'ils apparaissent dans le chunk i+1
        if fin_chunk_i.strip() in chunks[i + 1].page_content:
            has_overlap = True
            break

    assert has_overlap, "Aucun chevauchement détecté entre chunks consécutifs"
    print(" Chevauchement vérifié entre chunks consécutifs")
    print(" Test 3 réussi !\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("\n Tests de l'Étape 1 — Pipeline RAG basique\n")
    print("   (Tests locaux, pas de clé API requise)\n")

    try:
        test_split_documents()
        test_format_docs()
        test_overlap()

        print("=" * 60)
        print("🎉 Tous les tests de l'Étape 1 sont réussis !")
        print("=" * 60)
        print()
        print("Prochaines étapes pour le test complet :")
        print("   1. pip install -r requirements.txt")
        print("   2. Copier .env.example → .env et remplir les clés")
        print("   3. Déposer un PDF dans documents/")
        print("   4. python -m src.ingest")
        print("   5. python -m src.rag_chain")
        print()

    except AssertionError as e:
        print(f" Échec : {e}")
        exit(1)
