# Projet RAG — Assistant documentaire avec citations

Projet d'apprentissage LangChain : construire un assistant documentaire qui répond exclusivement à partir de PDF chargés par l'utilisateur, en citant ses sources.

## Stack

- **LangChain** — orchestration du pipeline RAG
- **Google Gemini** — modèle de langage (palier gratuit permanent, sans carte)
- **Voyage AI** — modèle d'embeddings (200M tokens gratuits par compte)
- **ChromaDB** — base vectorielle locale
- **Streamlit** — interface web

### Coût : zéro pour un projet d'apprentissage

| Service | Palier gratuit | Carte requise |
|---|---|---|
| Google Gemini | permanent, quotas par minute/jour | non |
| Voyage AI | 200M tokens par compte (permanent) | non |
| ChromaDB | local, illimité | — |

Obtenir les clés :
- Gemini : https://aistudio.google.com/apikey
- Voyage : https://dash.voyageai.com

Ordre de grandeur : un PDF de 50 pages ≈ 25 000 tokens à indexer. L'allocation
Voyage de 200M tokens couvre donc environ 8 000 PDF de cette taille.

Changer de fournisseur LLM ne demande que de remplacer `ChatGoogleGenerativeAI`
par `ChatAnthropic`, `ChatGroq`, etc. — le prompt, le retriever, la chaîne LCEL,
les citations et l'évaluation restent identiques. C'est tout l'intérêt de la
couche d'orchestration.

## Installation

Le projet s'installe dans un environnement virtuel dedie (`.venv/`), pour que ses
dependances n'aillent pas polluer le Python global de la machine.

```bash
cd projet-rag

# 1. Creer l'environnement virtuel
uv venv .venv --python 3.11
# equivalent sans uv : python -m venv .venv

# 2. L'activer
source .venv/Scripts/activate    # Windows (git-bash)
# .venv\Scripts\activate         # Windows (cmd / PowerShell)
# source .venv/bin/activate      # Linux / macOS

# 3. Installer les dependances DANS le venv
uv pip install -r requirements.txt
# equivalent sans uv : pip install -r requirements.txt

# 4. Configurer les cles API
cp .env.example .env
# Editer .env avec vos cles Anthropic et Voyage AI
```

Verifier qu'on travaille bien dans le venv :

```bash
python -c "import sys; print(sys.prefix)"
# doit afficher .../projet-rag/.venv
```

Sans activation, prefixer les commandes par l'interpreteur du venv :
`./.venv/Scripts/python.exe test_etape1.py`

### .venv vs .env — deux choses differentes

| | Role | Versionne ? |
|---|---|---|
| `.venv/` | dossier : interpreteur Python isole + paquets du projet | non (gitignore) |
| `.env` | fichier texte : cles API (`GOOGLE_API_KEY`, `VOYAGE_API_KEY`) | JAMAIS (gitignore) |

### Piege uv : toujours installer via requirements.txt

`uv pip install <un-seul-paquet>` **remplace** le contenu du venv au lieu d'y
ajouter : les autres paquets disparaissent. Pour ajouter une dependance,
l'inscrire dans `requirements.txt` puis relancer :

```bash
uv pip install -r requirements.txt
```

Verifier apres coup : `uv pip list | grep langchain` doit lister tous les paquets
langchain-*, pas seulement le dernier installe.

## Verification avant utilisation

```bash
python check_setup.py
```

A lancer apres l'installation et avant la premiere ingestion. C'est le
**seul** script qui fait de vrais appels reseau : il verifie l'interpreteur,
les 13 dependances, les cles du `.env`, puis appelle reellement les API
Voyage et Anthropic pour confirmer que les trois identifiants de modeles
existent encore.

Pourquoi c'est necessaire : `test_etape1..5.py` ne testent que la logique
locale (decoupage, citations, fusion RRF). Un nom de modele errone ou retire
passe tous ces tests sans broncher, et n'echoue qu'au premier appel reel.
Les ids Anthropic sont dates et sont retires periodiquement.

Le script sort en code 1 et s'arrete des la premiere etape bloquante, sans
gaspiller d'appels API. En cas d'echec, il affiche un diagnostic cible et le
lien vers la doc du fournisseur concerne.

## Utilisation

### Étape 1 — Pipeline CLI

```bash
# Placer des PDF dans documents/
python -m src.ingest
python -m src.rag_chain
```

### Étape 2 — Interface web

```bash
streamlit run src/app.py
```

## Architecture

```
src/
├── ingest.py       # Chargement PDF → découpage → indexation Chroma
├── rag_chain.py    # Chaîne RAG : retriever → prompt → LLM
├── memory.py       # Historique conversationnel multi-tours
└── app.py          # Interface Streamlit
```

## Étapes du projet

1. Pipeline RAG basique en CLI
2. Interface Streamlit + mémoire
3. Reformulation contextuelle des questions
4. Recherche hybride (sémantique + BM25)
5. Évaluation par jeu de questions-réference
