import re
import secrets
import string
import unicodedata
from pathlib import Path
from typing import List

import nltk
import spacy
from nltk.corpus import stopwords

from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from tensorflow.keras.models import load_model
import joblib

import pandas as pd

from configs.settings import log_cfg, api_cfg 
from utils.utils_logs import log_message


# =========================================================
# CONFIGURATIONS
# =========================================================
# --- CHEMINS
BASE_DIR = Path(__file__).resolve().parent.parent # /fake_news_nlp
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / api_cfg.api_ml_model_name
VECTORIZER_PATH = MODELS_DIR / api_cfg.api_ml_vectorizer_name

# --- SÉCURITÉ
API_KEY = api_cfg.api_key
USERNAME = api_cfg.api_user
PASSWORD = api_cfg.api_password

SECURITY = HTTPBasic()

# --- LOGS
FILE_LOG = log_cfg.file_log
FILE_LOG_DIR = BASE_DIR / log_cfg.log_dir_name
FILE_LOG_NAME = log_cfg.api_log_file_name

# =========================================================
# APPLICATION FASTAPI
# =========================================================
app = FastAPI(
    title=api_cfg.api_title,
    version=api_cfg.api_version,
    description=api_cfg.api_description
)

model = None
vectorizer = None


# =========================================================
# CHARGEMENT UNIQUE AU DEMARRAGE
# =========================================================
@app.on_event("startup")
def load_artifacts() -> None:

    global model, vectorizer

    try: 
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modele introuvable : {MODEL_PATH}")

        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(f"Vectoriseur introuvable : {VECTORIZER_PATH}")

        model = load_model(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        log_message(msg_log=f"API started/restarted successfully", file_log=FILE_LOG, file_log_dir=FILE_LOG_DIR, file_log_name=FILE_LOG_NAME)
    
    except Exception as e:
        log_message(level="error", msg_log=f"{e}", file_log=FILE_LOG, file_log_dir=FILE_LOG_DIR, file_log_name=FILE_LOG_NAME)



# =========================================================
# SCHEMAS PYDANTIC
# =========================================================
class PredictRequest(BaseModel):
    title: str


class BatchPredictRequest(BaseModel):
    titles: List[str]


class PredictResponse(BaseModel):
    title: str
    label: str
    confidence: float


class BatchPredictionItem(BaseModel):
    title: str
    label: str
    confidence: float


class BatchPredictResponse(BaseModel):
    predictions: List[BatchPredictionItem]

# =========================================================
# FONCTIONS DE SÉCURITÉ 
# =========================================================
def verify_api_key(x_api_key: str = Header(None)) -> None:
    
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Clé API manquante.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide.")

def verify_user(credentials: HTTPBasicCredentials = Depends(SECURITY)) -> None:
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)

    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Identifiants invalides.")
    
# =========================================================
# Fonctions utilitaires
# =========================================================
def validate_single_title(title: str) -> str:
    """
    Validation conforme à l enonce :
    - absent -> gere automatiquement par Pydantic/FastAPI => 422
    - vide / espaces -> 422
    - > 300 caracteres -> 400
    """
    if title is None:
        raise HTTPException(status_code=422, detail="Le champ  title  est obligatoire.")

    if not title.strip():
        raise HTTPException(
            status_code=422,
            detail="Le titre ne doit pas être vide ou compose uniquement d espaces."
        )

    if len(title) > 300:
        raise HTTPException(
            status_code=400,
            detail="Le titre ne doit pas depasser 300 caracteres."
        )

    return title


def validate_batch_titles(titles: List[str]) -> List[str]:
    """
    Validation conforme à l enonce :
    - champ absent -> gere automatiquement par FastAPI/Pydantic => 422
    - liste vide -> 400
    - plus de 50 titres -> 400
    - titre vide / espaces -> 422
    - titre > 300 caracteres -> 400
    """
    if titles is None:
        raise HTTPException(status_code=422, detail="Le champ  titles  est obligatoire.")

    if len(titles) == 0:
        raise HTTPException(
            status_code=400,
            detail="La liste des titres ne doit pas être vide."
        )

    if len(titles) > 50:
        raise HTTPException(
            status_code=400,
            detail="La liste des titres ne doit pas depasser 50 elements."
        )

    cleaned_titles = []
    for idx, title in enumerate(titles):
        if title is None:
            raise HTTPException(
                status_code=422,
                detail=f"Le titre à l index {idx} est manquant."
            )

        if not isinstance(title, str):
            raise HTTPException(
                status_code=422,
                detail=f"Le titre à l index {idx} doit être une chaîne de caracteres."
            )

        if not title.strip():
            raise HTTPException(
                status_code=422,
                detail=f"Le titre à l index {idx} est vide ou compose uniquement d espaces."
            )

        if len(title) > 300:
            raise HTTPException(
                status_code=400,
                detail=f"Le titre à l index {idx} depasse 300 caracteres."
            )

        cleaned_titles.append(title)

    return cleaned_titles

def sanitize_text(text) -> str:
    """Assainissement (& normalisation) du texte"""
    
    if text is None:
        return ""
    
    text = str(text)
    text = unicodedata.normalize("NFKD", text)

    # Apostrophes / guillemets
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')

    # Tirets Unicode
    text = text.replace("\u2010", "-")
    text = text.replace("\u2011", "-")
    text = text.replace("\u2012", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2015", "-")
    text = text.replace("\u2212", "-")
    text = text.replace("\x96", "-")

    # Espaces spéciaux
    text = text.replace("\xa0", " ")

    # Suppression des caractères de contrôle
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)

    # Conversion ASCII forcée
    text = text.encode("ascii", "ignore").decode("ascii")

    # Normalisation des espaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Téléchargements NLTK si nécessaire
nltk.download("stopwords")

# Chargement du modèle spaCy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Dictionnaire de contractions anglaises (20+)
CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "can't": "can not",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "i'm": "i am",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "isnt": "is not",
    "dont": "do not",
    "cant": "can not",
    "wont": "will not"
}

# Stopwords anglais NLTK, sauf négations
stop_words = set(stopwords.words("english"))
negations_to_keep = {"not", "no", "never", "neither"}
stop_words = stop_words - negations_to_keep

def expand_contractions(text: str, contractions: dict) -> str:
    """
    Remplace les contractions anglaises par leur forme développée.
    """
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, contractions.keys())) + r')\b')
    return pattern.sub(lambda x: contractions[x.group(0)], text)

def clean_text(text: str) -> str:
    """
    Nettoie un titre selon les étapes suivantes :
    1. minuscules
    2. suppression URLs et mentions
    3. suppression ponctuation et chiffres isolés
    4. expansion contractions
    5. suppression stopwords (sauf négations)
    6. lemmatisation spaCy
    7. suppression tokens de longueur < 2
    """
    if pd.isna(text):
        return ""

    # Conversion sécurisée en chaîne
    text = str(text)

    # 1. Mise en minuscules
    text = text.lower()

    # 2. Suppression des URLs et mentions
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    # 3. Suppression de la ponctuation et des chiffres isolés
    # On remplace la ponctuation par des espaces
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Suppression des chiffres isolés ou groupes de chiffres
    text = re.sub(r"\b\d+\b", " ", text)

    # 4. Expansion des contractions
    text = expand_contractions(text, CONTRACTIONS)

    # Normalisation des espaces avant spaCy
    text = re.sub(r"\s+", " ", text).strip()

    # 5 + 6 + 7 : stopwords, lemmatisation, suppression des petits tokens
    doc = nlp(text)

    cleaned_tokens = []
    for token in doc:
        lemma = token.lemma_.strip()

        # ignorer espaces / tokens vides
        if not lemma:
            continue

        # suppression stopwords sauf négations
        if lemma in stop_words:
            continue

        # suppression tokens longueur < 2
        if len(lemma) < 2:
            continue

        cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)

def predict_titles(titles_original: List[str], titles: List[str]) -> List[dict]:
    """
    Pipeline de prediction :
    texte -> TF-IDF -> modele Keras -> label/confidence
    """
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=500,
            detail="Le modele ou le vectoriseur n est pas charge."
        )

    X = vectorizer.transform(titles)
    X_dense = X.toarray()

    # Probabilite d être REAL (classe 1)
    probs_real = model.predict(X_dense, verbose=0).flatten()

    outputs = []
    for title_original, title, prob_real in zip(titles_original, titles, probs_real):
        label = "REAL" if prob_real > 0.5 else "FAKE"
        confidence = float(prob_real if label == "REAL" else 1 - prob_real)

        outputs.append({
            "title": title_original,
            "label": label,
            "confidence": round(confidence, 4)
        })

    return outputs


# =========================================================
# ENDPOINTS
# =========================================================
@app.get("/health")
async def health() -> dict:
    # log
    log_message(msg_log="Request: check API healt. Endpoint: /health", file_log=FILE_LOG, file_log_dir=FILE_LOG_DIR, file_log_name=FILE_LOG_NAME)

    return {"status": "ok", "model": "fake_news_detector"}

@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest,
            _: None = Depends(verify_api_key),
            __: None = Depends(verify_user)
) -> PredictResponse:
    try:
        title_input = validate_single_title(payload.title)

        # cleaning
        title = sanitize_text(title_input)
        title = clean_text(title)

        result = predict_titles([title_input], [title])[0]

        log_message(msg_log=f"Request: prediction on the text '{title_input}'. Endpoint: /predict", file_log=FILE_LOG, file_log_dir=FILE_LOG_DIR, file_log_name=FILE_LOG_NAME)
    
    except Exception as e:
        log_message(level="error", msg_log=f"Request: prediction on the text '{title_input}'. Endpoint: /predict. Error: {e}", file_log=FILE_LOG, file_log_dir=FILE_LOG_DIR, file_log_name=FILE_LOG_NAME)


    return PredictResponse(**result)

@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(payload: BatchPredictRequest,
                _: None = Depends(verify_api_key),
                __: None = Depends(verify_user) 
) -> BatchPredictResponse:
    try:
        titles_list = validate_batch_titles(payload.titles)

        # cleaning
        titles = [sanitize_text(t) for t in titles_list]
        titles = [clean_text(t) for t in titles]

        results = predict_titles(titles_list, titles)

        log_message(msg_log=f"Request: prediction on the texts list '{titles_list}'. Endpoint: /predict/batch", file_log=FILE_LOG, file_log_dir=FILE_LOG_DIR, file_log_name=FILE_LOG_NAME)
    
    except Exception as e:
        log_message(level="error", msg_log=f"Request: prediction on the text list '{titles_list}'. Endpoint: /predict. Error: {e}", file_log=FILE_LOG, file_log_dir=FILE_LOG_DIR, file_log_name=FILE_LOG_NAME)

    return BatchPredictResponse(
        predictions=[BatchPredictionItem(**item) for item in results]
    )