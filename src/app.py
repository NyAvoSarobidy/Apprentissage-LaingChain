"""
app.py — Interface Streamlit de l'assistant RAG
================================================
Interface web avec :
  - Zone de dépôt de PDF (upload)
  - Champ de question
  - Affichage des réponses avec sources
  - Historique de conversation (mémoire multi-tours)

Usage :
    streamlit run src/app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory import (
    init_memory,
    add_message,
    get_conversation_history,
    clear_memory,
)
from src.rag_chain import load_vectorstore, build_rag_chain, format_docs

# ---------------------------------------------------------------------------
# Chargement des cles API — DOIT venir avant tout appel aux modeles
# ---------------------------------------------------------------------------

# Streamlit n'a pas de fonction main() : le script s'execute de haut en bas
# a chaque interaction. Il faut donc appeler load_dotenv() au niveau module,
# et non dans une fonction comme dans ingest.py / rag_chain.py.
#
# Sans cet appel, os.getenv("VOYAGE_API_KEY") retourne None et
# VoyageAIEmbeddings leve :
#   ValueError: Must set `VOYAGE_API_KEY` environment variable
#
# On pointe explicitement vers le .env a la racine du projet : le repertoire
# de travail de Streamlit n'est pas garanti etre celui du projet.
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Verification immediate et lisible, plutot qu'un traceback pydantic
# au moment de cliquer sur "Indexer".
MISSING_KEYS = [
    name
    for name in ("GOOGLE_API_KEY", "VOYAGE_API_KEY")
    if not os.getenv(name)
]

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Assistant RAG",
    page_icon="📚",
    layout="wide",
)

# Si une cle manque, afficher un message actionnable et arreter net.
# st.stop() interrompt l'execution du script : plus propre qu'un traceback
# pydantic surgissant au clic sur "Indexer".
if MISSING_KEYS:
    st.error(
        f"Cle(s) API manquante(s) : {', '.join(MISSING_KEYS)}"
    )
    st.markdown(
        f"""
        Le fichier `.env` attendu est : `{ENV_PATH}`

        **Pour corriger :**
        ```bash
        cd projet-rag
        cp .env.example .env
        ```
        puis remplir les deux cles (gratuites, sans carte bancaire) :

        - `GOOGLE_API_KEY` → https://aistudio.google.com/apikey
        - `VOYAGE_API_KEY` → https://dash.voyageai.com

        Verifier ensuite avec :
        ```bash
        python check_setup.py
        ```
        """
    )
    st.stop()

# ---------------------------------------------------------------------------
# Barre latérale — Paramètres et upload
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 Assistant RAG")
    st.markdown("Assistant documentaire avec citations")

    # --- Upload de PDF ---
    st.markdown("---")
    st.subheader("📄 Documents")

    uploaded_files = st.file_uploader(
        "Dépose des PDF à indexer",
        type=["pdf"],
        accept_multiple_files=True,
        help="Les PDF seront ajoutés à la base vectorielle",
    )

    if uploaded_files:
        if st.button("🔄 Indexer les documents", use_container_width=True):
            with st.spinner("Indexation en cours..."):
                # Sauvegarder les fichiers uploadés dans documents/
                documents_dir = Path(__file__).parent.parent / "documents"
                documents_dir.mkdir(exist_ok=True)

                for uploaded_file in uploaded_files:
                    file_path = documents_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"✅ {uploaded_file.name} sauvegardé")

                # Lancer l'ingestion
                from src.ingest import load_pdfs, split_documents, index_in_chroma

                docs = load_pdfs(documents_dir)
                if docs:
                    chunks = split_documents(docs)
                    index_in_chroma(chunks)
                    st.success(f"✅ {len(chunks)} chunks indexés !")
                    st.cache_data.clear()
                    st.cache_resource.clear()

    # --- Bouton effacer historique ---
    st.markdown("---")
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        clear_memory(st.session_state)
        st.rerun()

    # --- Infos ---
    st.markdown("---")
    st.markdown(
        "**Fonctionnement :**\n"
        "- Dépose des PDF\n"
        "- Clique sur 'Indexer'\n"
        "- Pose des questions\n"
        "- Les réponses citent les sources"
    )

# ---------------------------------------------------------------------------
# Zone principale — Chat
# ---------------------------------------------------------------------------

st.title("💬 Pose une question sur tes documents")

# Initialiser la mémoire
init_memory(st.session_state)

# Afficher l'historique des messages
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# --- Saisie de la question ---
if prompt := st.chat_input("Ta question..."):
    # Afficher la question immédiatement
    with st.chat_message("user"):
        st.markdown(prompt)
    add_message(st.session_state, "human", prompt)

    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("Je cherche dans les documents..."):
            try:
                # Charger la base vectorielle
                vectorstore = load_vectorstore()

                if vectorstore is None:
                    response = (
                        "⚠️ Aucun document indexé. Dépose des PDF dans la barre "
                        "latérale et clique sur 'Indexer les documents'."
                    )
                else:
                    # Construire la chaîne RAG
                    rag_chain = build_rag_chain(vectorstore)

                    # Récupérer l'historique pour le contexte
                    history = get_conversation_history(st.session_state)

                    # Construire la question avec contexte
                    if history:
                        full_prompt = (
                            f"Historique de la conversation :\n{history}\n\n"
                            f"Nouvelle question : {prompt}"
                        )
                    else:
                        full_prompt = prompt

                    # Appeler la chaîne
                    response = rag_chain.invoke(full_prompt)

                # Afficher la réponse
                st.markdown(response)
                add_message(st.session_state, "ai", response)

            except Exception as e:
                error_msg = f"❌ Erreur : {str(e)}"
                st.error(error_msg)
                add_message(st.session_state, "ai", error_msg)

# --- Message d'accueil si vide ---
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Bonjour ! Je suis ton assistant documentaire.\n\n"
            "Pour commencer :\n"
            "1. **Dépose des PDF** dans la barre latérale\n"
            "2. **Clique sur 'Indexer'** pour les analyser\n"
            "3. **Pose-moi des questions** sur leur contenu\n\n"
            "Mes réponses citeront les sources : `[document.pdf, page N]`"
        )
