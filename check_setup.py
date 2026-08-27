"""
check_setup.py — Verification de l'installation avant l'ingestion
==================================================================
Script de "pre-vol" : verifie que tout est pret AVANT de lancer
l'ingestion complete (qui coute du temps et des appels API).

Ce script est le SEUL qui teste reellement le reseau. Les tests
test_etape1..5.py ne valident que la logique locale : ils ne peuvent
pas detecter un nom de modele errone ou une cle API invalide.

Ordre des verifications (du moins cher au plus cher) :
  1. Interpreteur : est-on bien dans le venv du projet ?
  2. Dependances : tous les paquets sont-ils importables ?
  3. Fichier .env : les cles sont-elles presentes et plausibles ?
  4. API Voyage : le modele d'embeddings repond-il ?
  5. API Anthropic : les deux modeles LLM repondent-ils ?
  6. Etat du projet : y a-t-il des PDF ? Chroma est-il coherent ?

Usage :
    python check_setup.py
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent

# Compteurs globaux de resultats
RESULTS = {"ok": 0, "warn": 0, "fail": 0}


# ---------------------------------------------------------------------------
# Helpers d'affichage
# ---------------------------------------------------------------------------


def section(title: str):
    """Affiche un titre de section."""
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def ok(message: str):
    """Verification reussie."""
    RESULTS["ok"] += 1
    print(f"   [OK]   {message}")


def warn(message: str):
    """Avertissement : non bloquant, mais a savoir."""
    RESULTS["warn"] += 1
    print(f"   [WARN] {message}")


def fail(message: str, hint: str = ""):
    """Echec bloquant."""
    RESULTS["fail"] += 1
    print(f"   [FAIL] {message}")
    if hint:
        print(f"          -> {hint}")


# ---------------------------------------------------------------------------
# 1. Interpreteur Python
# ---------------------------------------------------------------------------


def check_interpreter() -> bool:
    """
    Verifie qu'on tourne dans le venv du projet.

    Pourquoi c'est important : si on installe les dependances dans un
    venv mais qu'on lance le script avec un autre interpreteur, on
    obtient des ModuleNotFoundError incomprehensibles.

    On compare sys.prefix (l'environnement actif) au dossier .venv
    attendu a la racine du projet.
    """
    section("1. Interpreteur Python")

    expected_venv = (PROJECT_DIR / ".venv").resolve()
    current_prefix = Path(sys.prefix).resolve()

    print(f"   Version Python : {sys.version.split()[0]}")
    print(f"   Interpreteur   : {current_prefix}")

    # Version minimale : 3.11 (requise par le brief)
    if sys.version_info < (3, 11):
        fail(
            f"Python 3.11+ requis, trouve {sys.version_info.major}.{sys.version_info.minor}",
            "Recreer le venv : uv venv .venv --python 3.11",
        )
        return False

    if not expected_venv.exists():
        fail(
            "Le venv du projet (.venv/) n'existe pas",
            "uv venv .venv --python 3.11 && uv pip install -r requirements.txt",
        )
        return False

    if current_prefix == expected_venv:
        ok("Le script tourne bien dans le venv du projet")
        return True

    # Le venv existe mais n'est pas actif : avertissement, pas echec,
    # car les imports peuvent quand meme fonctionner par ailleurs.
    warn("Le venv du projet existe mais n'est PAS l'interpreteur actif")
    print(f"          attendu : {expected_venv}")
    print(f"          actif   : {current_prefix}")
    print("          -> source .venv/Scripts/activate  (git-bash)")
    print("          -> ou    ./.venv/Scripts/python.exe check_setup.py")
    return True


# ---------------------------------------------------------------------------
# 2. Dependances
# ---------------------------------------------------------------------------

# (module importable, nom du paquet pip, role dans le projet)
DEPENDENCIES = [
    ("langchain_core", "langchain-core", "primitives LCEL (prompts, runnables)"),
    ("langchain_google_genai", "langchain-google-genai", "appel aux modeles Gemini"),
    ("langchain_community", "langchain-community", "loaders PDF + BM25Retriever"),
    ("langchain_text_splitters", "langchain-text-splitters", "decoupage en chunks"),
    ("langchain_chroma", "langchain-chroma", "connecteur Chroma"),
    ("chromadb", "chromadb", "base vectorielle locale"),
    ("langchain_voyageai", "langchain-voyageai", "embeddings Voyage"),
    ("voyageai", "voyageai", "SDK Voyage"),
    ("rank_bm25", "rank_bm25", "moteur BM25 (Etape 4)"),
    ("pypdf", "pypdf", "lecture des fichiers PDF"),
    ("streamlit", "streamlit", "interface web (Etape 2)"),
    ("dotenv", "python-dotenv", "chargement du fichier .env"),
    ("tqdm", "tqdm", "barres de progression"),
]


def check_dependencies() -> bool:
    """
    Verifie que chaque dependance est importable.

    Piege classique : rank_bm25. L'import de BM25Retriever depuis
    langchain_community reussit meme sans rank_bm25 installe --
    l'erreur ne surgit qu'a l'appel de .from_documents(). D'ou
    l'importance de tester l'import du paquet lui-meme ici.
    """
    section("2. Dependances")

    missing = []
    for module_name, package_name, role in DEPENDENCIES:
        try:
            __import__(module_name)
            ok(f"{package_name:<28} {role}")
        except ImportError:
            fail(f"{package_name:<28} MANQUANT ({role})")
            missing.append(package_name)

    if missing:
        print()
        print("   Installer les paquets manquants :")
        print(f"     uv pip install {' '.join(missing)}")
        print("   ou tout reinstaller :")
        print("     uv pip install -r requirements.txt")
        return False

    return True


# ---------------------------------------------------------------------------
# 3. Fichier .env et cles API
# ---------------------------------------------------------------------------


def mask(secret: str) -> str:
    """
    Masque une cle API pour un affichage sans fuite.

    On ne montre que le prefixe et les 4 derniers caracteres :
    assez pour reconnaitre la cle, pas assez pour la reutiliser.
    Ne JAMAIS print() une cle en clair, meme dans un script local :
    les logs de terminal se partagent par copier-coller.
    """
    if len(secret) <= 12:
        return "*" * len(secret)
    return f"{secret[:7]}...{secret[-4:]} ({len(secret)} car.)"


def check_env_file() -> bool:
    """
    Verifie la presence du .env et la plausibilite des cles.

    On teste ici le FORMAT, pas la validite : une cle bien formee peut
    etre revoquee. La validite reelle est testee aux etapes 4 et 5,
    qui font un vrai appel reseau.
    """
    section("3. Fichier .env et cles API")

    env_path = PROJECT_DIR / ".env"
    example_path = PROJECT_DIR / ".env.example"

    if not env_path.exists():
        fail(
            "Le fichier .env est absent",
            f"cp {example_path.name} .env  puis remplir les deux cles",
        )
        return False

    ok(".env present")

    # Charger sans ecraser d'eventuelles variables deja exportees
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)

    all_good = True

    # --- Cle Google (Gemini) ---
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not google_key:
        fail("GOOGLE_API_KEY absente ou vide")
        all_good = False
    elif "xxxx" in google_key or google_key.startswith("AIzaSyXXX"):
        fail(
            "GOOGLE_API_KEY contient encore la valeur d'exemple",
            "Remplacer par la vraie cle depuis aistudio.google.com/apikey",
        )
        all_good = False
    elif not google_key.startswith("AIza"):
        warn(f"GOOGLE_API_KEY ne commence pas par 'AIza' : {mask(google_key)}")
    else:
        ok(f"GOOGLE_API_KEY : {mask(google_key)}")

    # --- Cle Voyage ---
    voyage_key = os.getenv("VOYAGE_API_KEY", "").strip()
    if not voyage_key:
        fail("VOYAGE_API_KEY absente ou vide")
        all_good = False
    elif voyage_key.startswith("pa-xxx") or "xxxx" in voyage_key:
        fail(
            "VOYAGE_API_KEY contient encore la valeur d'exemple",
            "Remplacer par la vraie cle depuis dash.voyageai.com",
        )
        all_good = False
    elif not voyage_key.startswith("pa-"):
        warn(f"VOYAGE_API_KEY ne commence pas par 'pa-' : {mask(voyage_key)}")
    else:
        ok(f"VOYAGE_API_KEY : {mask(voyage_key)}")

    # --- Garde-fou anti-commit ---
    gitignore = PROJECT_DIR / ".gitignore"
    if gitignore.exists():
        contenu = gitignore.read_text(encoding="utf-8")
        if ".env" in contenu:
            ok(".env est bien ignore par git")
        else:
            fail(
                ".env n'est PAS dans .gitignore -- risque de commit des cles",
                "Ajouter une ligne '.env' dans .gitignore",
            )
            all_good = False

    return all_good


# ---------------------------------------------------------------------------
# 4. API Voyage — le modele d'embeddings existe-t-il vraiment ?
# ---------------------------------------------------------------------------


def check_voyage_api() -> bool:
    """
    Fait un VRAI appel a l'API Voyage sur une phrase minuscule.

    C'est la verification la plus importante du script : c'est la seule
    qui peut detecter un nom de modele inexistant. Un mauvais nom ne
    provoque aucune erreur a l'import ni au chargement de la config --
    seulement au moment de l'appel.

    On verifie aussi la DIMENSION du vecteur retourne, car c'est elle
    qui doit correspondre a ce que contient Chroma.
    """
    section("4. API Voyage (embeddings)")

    from src.rag_chain import EMBEDDING_MODEL

    print(f"   Modele teste : {EMBEDDING_MODEL}")
    print("   Appel reseau en cours (1 phrase, cout negligeable)...")

    try:
        from langchain_voyageai import VoyageAIEmbeddings

        embeddings = VoyageAIEmbeddings(model=EMBEDDING_MODEL)
        vector = embeddings.embed_query("Ceci est un test de configuration.")

        if not vector:
            fail("L'API a repondu mais le vecteur est vide")
            return False

        ok(f"Modele '{EMBEDDING_MODEL}' valide")
        ok(f"Dimension du vecteur : {len(vector)}")
        print(f"          -> cette dimension doit rester identique entre")
        print(f"             l'indexation (ingest.py) et l'interrogation.")
        return True

    except Exception as exc:
        message = str(exc)
        fail(f"Appel Voyage echoue : {type(exc).__name__}")
        print(f"          {message[:250]}")

        # Diagnostic cible selon le message d'erreur
        low = message.lower()
        if "model" in low and ("not found" in low or "invalid" in low or "does not exist" in low):
            print()
            print("          DIAGNOSTIC : le nom du modele est invalide ou retire.")
            print("          Modeles Voyage actuels :")
            print("            voyage-4        generaliste (defaut du projet)")
            print("            voyage-4-large  meilleure qualite")
            print("            voyage-4-lite   moins cher / plus rapide")
            print("            voyage-law-2    domaine juridique")
            print("            voyage-finance-2 domaine finance")
            print("          Verifier : https://docs.voyageai.com/docs/embeddings")
            print("          Corriger EMBEDDING_MODEL dans src/ingest.py ET src/rag_chain.py")
        elif "auth" in low or "401" in low or "unauthorized" in low or "api key" in low:
            print()
            print("          DIAGNOSTIC : cle API refusee.")
            print("          Verifier VOYAGE_API_KEY sur https://dash.voyageai.com")
        elif "quota" in low or "429" in low or "rate" in low:
            print()
            print("          DIAGNOSTIC : quota depasse ou trop d'appels.")
            print("          Attendre un peu, ou verifier le plan Voyage.")
        return False


# ---------------------------------------------------------------------------
# 5. API Anthropic — les deux modeles LLM repondent-ils ?
# ---------------------------------------------------------------------------


def extraire_texte(reponse) -> str:
    """
    Extrait le texte d'une reponse de LLM, quel que soit son format.

    Les modeles recents renvoient un `content` qui peut etre :
      - une chaine simple : "OK"
      - une liste de blocs : [{"type": "text", "text": "OK", ...}]

    Le second format est celui des modeles a raisonnement, qui melangent
    blocs de texte et blocs internes. Cette fonction normalise les deux.

    Note : dans une chaine LCEL, StrOutputParser() fait ce travail
    automatiquement. Cette fonction n'est necessaire que lorsqu'on appelle
    .invoke() directement sur le modele, comme ici.
    """
    contenu = getattr(reponse, "content", reponse)

    if isinstance(contenu, str):
        return contenu

    if isinstance(contenu, list):
        morceaux = []
        for bloc in contenu:
            if isinstance(bloc, dict) and "text" in bloc:
                morceaux.append(str(bloc["text"]))
            elif isinstance(bloc, str):
                morceaux.append(bloc)
        return "".join(morceaux)

    return str(contenu)


def check_gemini_api() -> bool:
    """
    Teste les DEUX modeles Gemini utilises par le projet.

    Le projet en utilise deux, et il faut valider les deux separement :
      - LLM_MODEL         : redige la reponse citee (Flash)
      - REFORMULATOR_MODEL: reecrit les questions de suivi (Flash-Lite)

    Piege : preferer les ids "stable" aux ids "-preview". Les previews sont
    deprecies avec seulement 2 semaines de preavis, et les modeles retires
    renvoient une erreur.
    """
    section("5. API Google Gemini (LLM)")

    from src.rag_chain import LLM_MODEL, REFORMULATOR_MODEL

    from langchain_google_genai import ChatGoogleGenerativeAI

    all_good = True

    for label, model_name in [
        ("LLM principal ", LLM_MODEL),
        ("Reformulateur ", REFORMULATOR_MODEL),
    ]:
        print(f"   {label}: {model_name}")
        try:
            # PIEGE VERIFIE : ne PAS mettre max_output_tokens tres bas.
            # Les modeles Gemini 3.x sont des modeles a raisonnement : ils
            # consomment des tokens internes de reflexion AVANT de produire du
            # texte. Avec max_output_tokens=10, le budget est epuise pendant la
            # reflexion et content revient vide ([]) alors que l'appel a reussi.
            # On laisse donc le defaut du modele.
            #
            # temperature n'est pas passe non plus : les modeles Gemini 3.x
            # utilisent un sampling fixe et emettent un UserWarning si on tente
            # de le regler.
            llm = ChatGoogleGenerativeAI(model=model_name)
            reponse = llm.invoke("Reponds uniquement par le mot: OK")

            # content peut etre une chaine OU une liste de blocs structures
            # [{'type': 'text', 'text': 'OK', ...}]. On normalise.
            texte = extraire_texte(reponse)

            if not texte.strip():
                fail(f"'{model_name}' a repondu, mais le contenu est vide")
                print("          L'appel a reussi (cle et modele valides) mais")
                print("          aucun texte n'a ete produit. Cause probable :")
                print("          budget max_output_tokens trop bas pour un modele")
                print("          a raisonnement.")
                all_good = False
                continue

            ok(f"'{model_name}' repond -> {texte.strip()[:30]}")

        except Exception as exc:
            message = str(exc)
            fail(f"'{model_name}' inutilisable : {type(exc).__name__}")
            print(f"          {message[:250]}")

            low = message.lower()
            if "not found" in low or "404" in low or "is not supported" in low:
                print()
                print("          DIAGNOSTIC : identifiant de modele invalide ou RETIRE.")
                print("          Modeles Gemini stables actuels :")
                print("            gemini-3.7-flash        le plus capable")
                print("            gemini-3.6-flash        defaut du projet")
                print("            gemini-3.5-flash-lite   rapide (reformulateur)")
                print("            gemini-3.1-flash-lite   le moins cher")
                print("          Eviter les ids '-preview' : 2 semaines de preavis.")
                print("          Liste a jour : https://ai.google.dev/gemini-api/docs/models")
                print("          Corriger dans src/rag_chain.py ET eval/evaluate.py")
            elif "api key" in low or "401" in low or "permission" in low or "invalid" in low:
                print()
                print("          DIAGNOSTIC : cle API refusee.")
                print("          Verifier GOOGLE_API_KEY sur aistudio.google.com/apikey")
            elif "quota" in low or "429" in low or "resource_exhausted" in low:
                print()
                print("          DIAGNOSTIC : quota du palier gratuit atteint.")
                print("          Les quotas RPD se remettent a zero a minuit heure Pacifique.")
                print("          Attendre, ou utiliser un modele -flash-lite (quota separe).")
            all_good = False

    return all_good


# ---------------------------------------------------------------------------
# 6. Etat du projet (PDF et base Chroma)
# ---------------------------------------------------------------------------


def check_project_state() -> bool:
    """
    Verifie les donnees du projet : PDF sources et base vectorielle.

    Aucun appel reseau ici. On signale surtout l'incoherence classique :
    une base Chroma peuplee alors que documents/ est vide, ou l'inverse.
    """
    section("6. Etat du projet")

    documents_dir = PROJECT_DIR / "documents"
    chroma_dir = PROJECT_DIR / "chroma_db"

    # --- PDF sources ---
    pdfs = sorted(documents_dir.glob("*.pdf")) if documents_dir.exists() else []
    if pdfs:
        ok(f"{len(pdfs)} PDF dans documents/")
        for pdf in pdfs[:5]:
            taille_ko = pdf.stat().st_size / 1024
            print(f"          - {pdf.name} ({taille_ko:.0f} Ko)")
        if len(pdfs) > 5:
            print(f"          ... et {len(pdfs) - 5} autre(s)")
    else:
        warn("Aucun PDF dans documents/")
        print("          -> deposer au moins un PDF avant python -m src.ingest")

    # --- Base Chroma ---
    # On ne compte pas les chunks ici : ouvrir la collection instancierait
    # le modele d'embeddings et declencherait un appel reseau de plus.
    # On se contente de regarder si le dossier contient des donnees.
    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        fichiers = list(chroma_dir.rglob("*"))
        ok(f"Base Chroma presente ({len(fichiers)} fichiers)")
        if not pdfs:
            warn("Chroma contient des donnees mais documents/ est vide")
            print("          -> l'index reference des PDF qui ne sont plus la.")
            print("             BM25 (Etape 4) relit documents/ a chaque")
            print("             lancement : il sera vide, donc la recherche")
            print("             hybride retombera sur le vectoriel seul.")
    else:
        warn("Base Chroma vide -- lancer : python -m src.ingest")

    # Cette section n'est jamais bloquante : elle informe seulement.
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """
    Enchaine les verifications de la moins couteuse a la plus couteuse.

    On s'arrete des qu'une etape bloquante echoue : inutile d'appeler
    les API si les dependances manquent ou si les cles sont absentes.
    """
    print()
    print("#" * 64)
    print("#  VERIFICATION DE L'INSTALLATION — projet RAG")
    print("#" * 64)

    # Permettre les imports 'from src...' meme sans installation en paquet
    sys.path.insert(0, str(PROJECT_DIR))

    # --- Etapes locales (gratuites, bloquantes) ---
    check_interpreter()

    if not check_dependencies():
        conclure(bloque_a="les dependances")
        return 1

    if not check_env_file():
        conclure(bloque_a="la configuration des cles")
        return 1

    # --- Etapes reseau (facturees, bloquantes) ---
    voyage_ok = check_voyage_api()
    anthropic_ok = check_gemini_api()

    # --- Etat des donnees (informatif) ---
    check_project_state()

    if not (voyage_ok and anthropic_ok):
        conclure(bloque_a="les appels API")
        return 1

    conclure()
    return 0


def conclure(bloque_a: str = ""):
    """Affiche le bilan final et la suite a donner."""
    section("BILAN")
    print(f"   Reussis        : {RESULTS['ok']}")
    print(f"   Avertissements : {RESULTS['warn']}")
    print(f"   Echecs         : {RESULTS['fail']}")
    print()

    if bloque_a:
        print(f"   ARRET : corriger {bloque_a} avant de continuer.")
        print("   Relancer ensuite : python check_setup.py")
        return

    print("   Installation validee. Les deux API repondent et les")
    print("   identifiants de modeles sont corrects.")
    print()
    print("   Suite :")
    print("     1. python -m src.ingest      indexer les PDF")
    print("     2. python -m src.rag_chain   poser des questions en CLI")
    print("     3. streamlit run src/app.py  interface web")
    print("     4. python -m eval.evaluate   mesurer la qualite")


if __name__ == "__main__":
    sys.exit(main())
