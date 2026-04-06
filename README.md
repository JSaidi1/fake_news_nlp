# Fake News Detection NLP

## Contexte

Projet réalisé dans le cadre d’un ECF en traitement automatique du langage naturel (NLP).

*Dans un contexte de ``fact-checking``, un organisme reçoit un grand volume de titres d’articles provenant de sources variées. Certains sont fiables, d’autres sont trompeurs ou sensationnalistes. L’objectif est d’automatiser un premier niveau de tri afin de réduire la charge de travail des analystes humains.*

*Ce projet consiste à concevoir un pipeline NLP complet permettant de ``classifier automatiquement`` les titres en deux catégories : **REAL** ou **FAKE**.*

---

## Objectifs

- Nettoyage et prétraitement des données textuelles
- Analyse exploratoire du corpus
- Vectorisation (TF-IDF et embeddings)
- Entraînement de modèles de classification
- Évaluation des performances
- Déploiement du modèle via une API REST sécurisée

---
## Organisation du projet

Le projet est structuré en deux parties principales :

### 1. Étude et modélisation (Notebook)

Le notebook `ecf_fake_news.ipynb` regroupe l’ensemble du travail d’analyse et de modélisation :

- Analyse exploratoire des données
- Prétraitement des textes
- Vectorisation (TF-IDF et embeddings)
- Entraînement des modèles
- Évaluation des performances
- Enregistrement des modèles produits pour la future exploitation par l'API

---

### 2. API de prédiction

Le dossier `api/` contient une API REST développée avec FastAPI permettant d’exposer le modèle en production :

- Endpoint pour vérifier que l’API est bien démarrée (`/health`)
- Endpoint de prédiction simple (`/predict`)
- Endpoint de prédiction en batch (`/predict/batch`)
- Endpoint Swagger qui fournit automatiquement une interface pour décrire et tester l’API (`/docs`)
- Sécurisation via API Key et authentification (basique)


## Modèles implémentés

Deux approches ont été développées :

### 1. Modèle baseline
- TF-IDF
- Réseau de neurones dense
- Accuracy ~79%

### 2. Modèle avancé (retenu)
- TextVectorization + Embedding
- LSTM bidirectionnel
- Accuracy ~82%
- AUC ~0.89

Le modèle LSTM a été retenu pour ses meilleures performances globales. (*l'API aura comme modèle le Modèle baseline*).

---

## Performances

| Modèle | Accuracy | AUC |
|--------|--------|-----|
| TF-IDF Dense | ~0.79 | ~0.86 |
| LSTM Bidirectionnel | ~0.82 | ~0.89 |

---
## Pipeline NLP
    Titre brut
    ↓
    Prétraitement (clean_title)
    ↓
    Vectorisation (TF-IDF / TextVectorization)
    ↓
    Modèle (Dense / LSTM)
    ↓
    Prédiction (REAL / FAKE + score)
---

## Limites

Le modèle apprend des **patterns linguistiques spécifiques au dataset** et peut rencontrer des difficultés sur des titres neutres ou hors distribution. Cela reflète les limites des modèles supervisés en NLP.

---

## Structure du projet
    fake_news_nlp
    │
    │   .env.example
    │   .gitignore
    │   .python-version
    │   pyproject.toml
    │   README.md
    │   uv.lock
    │   
    ├───api
    │       main.py
    │       
    ├───configs
    │       settings.py
    │       
    ├───data
    │       fake_or_real_news.csv
    │       fake_or_real_news_original.csv
    │       titles_clean.csv
    │       
    ├───docs
    │       ECF_4_NLP_TF_FakeNews.md
    │       
    ├───logs
    │       api.log
    │       
    ├───models
    │       best_model_lstm.keras
    │       best_model_tfidf.keras
    │       tfidf_vectorizer.pkl
    │       
    ├───notebook
    │       ecf_fake_news.ipynb
    │       
    ├───output
    │       graph_dist_length_titles.png
    │       graph_loss_acc_lstm.png
    │       graph_loss_acc_tfidf.png
    │       graph_mat_conf_lstm.png
    │       graph_roc_lstm.png
    │       titles_pretreated.csv
    │       
    └───utils
            utils_logs.py

## Configuration (fichier .env)

Le projet utilise un fichier de configuration `.env` pour centraliser les paramètres de l’application (API, sécurité, modèle, logs).

Un fichier `.env.example` est fourni comme modèle.

## Installation et utilisation

### Dépendances
Le projet utilise `pyproject.toml` pour la gestion des dépendances.  
Toutes les bibliothèques nécessaires y sont définies.

#### Principales technologies

- **FastAPI** — API REST
- **TensorFlow / Keras** — Modélisation
- **scikit-learn** — TF-IDF et évaluation
- **pandas / numpy** — Manipulation de données
- **joblib** — Sérialisation des modèles

---

#### Installation des dépendances avec uv (recommandé)

```bash
uv sync
```

#### Installation des dépendances avec pip
```bash
pip install .
```

### Partie étude et modélisation (Notebook)

Le notebook `ecf_fake_news.ipynb` regroupe l’ensemble du travail d’analyse et de modélisation : Il suffit de l'executer et il produira tous les outputs (graphiques ...) ainsi que les modèles entrainés.

### Partie API

Pour utiliser l'API, il faut :

- Se placer à la racine du projet et activer la l'environement virtuel (si on veut travailler dedans) :

        .\.venv\Scripts\activate

- Avoir les modèles dans /models (si les modèles ne sont pas disponibles, il suffit d'executer le notebook `ecf_fake_news.ipynb` et il chargera les modèles nécéssaires)

- La démarrer. Commande : 

        uvicorn api.main:app --reload


N.B : 
- dès le démarrage de l'API, les ``logs`` s'enregistre dans le fichier ``/logs/api.log``
- Endpoint Swagger qui fournit automatiquement une interface pour décrire et tester l’API (/docs)


## Auteur
J.SAIDI
(06/04/2026)