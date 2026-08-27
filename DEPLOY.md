# Guide de déploiement — Streamlit Community Cloud

## Pourquoi Streamlit Community Cloud ?

- **Gratuit** pour les projets publics
- **1 clic** depuis GitHub
- Supporte nativement Streamlit (contrairement à Vercel)
- Gère les secrets de manière sécurisée

## Étapes de déploiement

### 1. Prérequis

- [ ] Le repo GitHub est à jour (`git push`)
- [ ] `requirements.txt` est à la racine du projet (ou dans `projet-rag/`)
- [ ] `src/app.py` est le point d'entrée

### 2. Créer le compte

1. Va sur https://streamlit.io/cloud
2. Clique sur « Sign up with GitHub »
3. Autorise Streamlit à accéder à tes repos

### 3. Déployer l'application

1. Clique sur « New app »
2. Sélectionne le repo `Apprentissage-LaingChain`
3. **Important** : dans « Main file path », mets `projet-rag/src/app.py`
4. Clique sur « Deploy! »

### 4. Configurer les secrets

Une fois l'app créée :

1. Clique sur « Settings » (icône engrenage) → « Secrets »
2. Ajoute les deux cles :

```toml
GOOGLE_API_KEY = "AIzaSy..."
VOYAGE_API_KEY = "pa-..."
```

3. Clique sur « Save »
4. Redémarre l'app (« Reboot »)

### 5. Vérifier le déploiement

- L'app est accessible à `https://<app-name>.streamlit.app`
- Le premier chargement prend 2-3 minutes (installation des dépendances)
- Vérifie les logs dans « Settings » → « Logs » en cas d'erreur

## Obtenir les clés API

### Google Gemini (gratuit, sans carte)

1. Va sur https://aistudio.google.com/apikey
2. Clique sur « Create API key »
3. Copie la clé (commence par `AIza...`)

### Voyage AI (200M tokens gratuits, sans carte)

1. Va sur https://dash.voyageai.com
2. Crée un compte
3. Va dans « API Keys » → « Create Key »
4. Copie la clé (commence par `pa-...`)

## Résolution des problèmes

### « ModuleNotFoundError »

Vérifie que `requirements.txt` contient toutes les dépendances nécessaires.

### « API key not found »

Vérifie que les secrets sont bien configurés dans le dashboard Streamlit.

### « RateLimitError »

Voyage limite le débit sans carte à 3 req/min. L'ingestion prend quelques minutes — c'est normal.

### L'app ne répond pas

Vérifie les logs dans le dashboard Streamlit pour identifier l'erreur.

## Alternative : déploiement local

```bash
cd projet-rag
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sur Windows
pip install -r requirements.txt
streamlit run src/app.py
```

L'app est accessible à `http://localhost:8501`.
