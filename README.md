# 🩸 HémoVision — Prédiction de Succès de Greffe de Moelle Osseuse Pédiatrique

> Projet Coding Week · Mars 2026 · Équipe : Jay · Léandre Zadi · Adama Sana · Ilias Janati

---

## 📌 Présentation

HémoVision est une application d'aide à la décision médicale qui prédit la **survie d'un enfant après une greffe allogénique de cellules souches hématopoïétiques** (bone marrow transplant). Le modèle est entraîné sur 187 patients pédiatriques atteints de maladies hématologiques graves (LAL, LAM, aplasie médullaire, etc.).

L'interface Streamlit permet à un praticien de saisir les données pré-opératoires d'un patient et d'obtenir instantanément une prédiction accompagnée d'une explication SHAP.

---

## 🗂️ Structure du Projet

```
bone_marrow_project/
│
├── data/
│   ├── bone-marrow.arff        # Dataset source (Silesian University)
│   ├── X_test.csv              # Données de test (générées par train_model.py)
│   └── y_test.csv              # Labels de test
│
├── models/
│   ├── xgboost_model.pkl       # Pipeline XGBoost (StandardScaler + XGBClassifier)
│   ├── randomforest_model.pkl  # Pipeline Random Forest
│   ├── svm_model.pkl           # Pipeline SVM
│   └── features_list.pkl       # Liste des 44 features finales
│
├── data_processing.py          # Pipeline de prétraitement complet
├── train_model.py              # Entraînement et sauvegarde des modèles
├── evaluate_model.py           # Évaluation et visualisations
├── app.py                      # Interface Streamlit (HémoVision)
├── eda.md                      # Analyse exploratoire des données
└── README.md                   # Ce fichier
```

---

## ⚙️ Installation

### Prérequis

- Python 3.10+
- pip

### Dépendances

```bash
pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn shap scipy matplotlib seaborn joblib
```

---

## 🚀 Lancement

### 1. Placer le dataset

Copier `bone-marrow.arff` dans le dossier `data/`.

### 2. Entraîner les modèles

```bash
python train_model.py
```

Génère les fichiers `.pkl` dans `models/` et les données de test dans `data/`.

### 3. Évaluer les modèles

```bash
python evaluate_model.py
```

Génère les visualisations dans `outputs/` : matrices de confusion, courbes ROC, feature importances, heatmap des métriques.

### 4. Lancer l'interface

```bash
streamlit run app.py
```

---

## 🔬 Pipeline ML

```
bone-marrow.arff
      │
      ▼
load_arff()          ← Décodage bytes, remplacement des '?' par NaN
      │
      ▼
impute()             ← KNN (numérique) + Mode (catégoriel)
      │
      ▼
optimize_memory()    ← float64→float32 (après imputation)
      │
      ▼
prepare_target()     ← survival_status → target (0=survie, 1=décès)
      │
      ▼
encode_split_resample()
      ├─ One-Hot Encoding (drop_first=True)
      ├─ Train/Test split stratifié (80/20)
      └─ SMOTE sur le train uniquement
      │
      ▼
drop_invalid_cols()
      ├─ 8 colonnes post-greffe (leakage)
      └─ 5 colonnes redondantes (binarisations)
      │
      ▼
Pipeline sklearn (StandardScaler + Estimateur)
      ├─ RandomForestClassifier
      ├─ XGBClassifier
      └─ SVC (probability=True)
```

---

## 🚫 Data Leakage — Colonnes Exclues

### Colonnes post-greffe (non disponibles au moment de la décision)

| Colonne | Raison |
|---|---|
| `IIIV` | GvHD aiguë stade II/III/IV — observable après greffe |
| `Relapse` | Rechute de la maladie — après greffe |
| `aGvHDIIIIV` | GvHD aiguë stade III/IV — après greffe |
| `extcGvHD` | GvHD chronique extensive — après greffe |
| `ANCrecovery` | Temps récupération neutrophiles — après greffe |
| `PLTrecovery` | Temps récupération plaquettes — après greffe |
| `time_to_aGvHD_III_IV` | Délai avant GvHD III/IV — après greffe |
| `survival_time` | Durée de survie — après greffe |

### Colonnes redondantes (multicolinéarité)

| Supprimée | Conservée |
|---|---|
| `Donorage35` | `Donorage` |
| `Recipientage10` | `Recipientage` |
| `Recipientageint` | `Recipientage` |
| `HLAmismatch` | `HLAmatch` |
| `Diseasegroup` | `Disease` |

---

## 📊 Résultats (sur le jeu de test, 38 patients)

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.6316 | 0.6154 | 0.4706 | 0.5333 | **0.7255** |
| Random Forest | 0.6316 | 0.6154 | 0.4706 | 0.5333 | 0.6751 |
| SVM | 0.5789 | 1.0000 | 0.0588 | 0.1111 | 0.6303 |

> **Modèle sélectionné : XGBoost** — meilleur ROC-AUC (0.7255) sur 38 patients de test (21 survies, 17 décès). Le SVM présente une précision parfaite mais un recall quasi nul (0.059), signe d'un modèle trop conservateur inutilisable en clinique.

> Les métriques sont générées après entraînement par `evaluate_model.py` et sauvegardées dans `outputs/model_comparison.csv`.

---

## 🧬 Explicabilité SHAP

Le modèle XGBoost est accompagné d'une explication SHAP (SHapley Additive exPlanations) pour chaque prédiction individuelle. Le graphique **waterfall** décompose la prédiction en contributions de chaque variable, permettant au praticien de comprendre *pourquoi* le modèle prédit un risque élevé ou faible.

---

## 🤖 Prompt Engineering

Cette section documente comment nous avons utilisé Claude (Anthropic) pour accélérer le développement du projet, conformément aux exigences pédagogiques.

---

### Tâche sélectionnée : Implémentation de `data_processing.py`

Nous avons choisi de documenter la conception du pipeline de prétraitement, qui était la tâche la plus critique et la plus complexe du projet.

---

### Prompt initial (version naïve)

> *"Écris une fonction Python pour charger un fichier ARFF, gérer les valeurs manquantes et préparer les données pour un modèle de machine learning."*

**Résultat obtenu :** Une fonction monolithique de 40 lignes mélangeant chargement, imputation, encodage et split. Elle ne gérait pas les '?' ARFF, utilisait `fillna(0)` pour toutes les colonnes, et appliquait le StandardScaler avant le KNN (ce qui est incorrect).

**Problèmes identifiés :**
- Les `?` du format ARFF non remplacés → créaient de fausses catégories après OHE
- KNN sur données brutes (non normalisées) → distances biaisées par les unités
- SMOTE appliqué sur tout le dataset → data leakage sur le test set
- `optimize_memory()` appliqué avant imputation → float32 dégradait la précision du KNN

---

### Prompt amélioré (version finale)

> *"Écris un pipeline de prétraitement Python modulaire pour le dataset BMT Children (format ARFF, 187 patients, 37 variables). Contraintes :*
> *1. Remplacer les '?' ARFF par np.nan dès le chargement (scipy.io.arff)*
> *2. Imputer séparément : KNNImputer sur les colonnes numériques APRÈS StandardScaler + inverse_transform, SimpleImputer(mode) sur les catégorielles*
> *3. Appliquer optimize_memory() (float64→float32) APRÈS l'imputation pour préserver la précision KNN*
> *4. SMOTE uniquement sur le train set (pas de leakage)*
> *5. Chaque étape dans une fonction séparée avec docstring expliquant le 'pourquoi'"*

**Résultat obtenu :** Le pipeline modulaire en 6 fonctions (`load_arff`, `impute`, `optimize_memory`, `prepare_target`, `encode_split_resample`, `save_test_data`) avec docstrings détaillées justifiant chaque choix technique.

---

### Analyse de l'efficacité

| Critère | Prompt naïf | Prompt amélioré |
|---|---|---|
| Gestion des `?` ARFF | ❌ Non géré | ✅ `replace("?", np.nan)` |
| Ordre KNN / normalisation | ❌ Inversé | ✅ Scale → KNN → inverse |
| Ordre KNN / optimize_memory | ❌ Inversé | ✅ Impute d'abord |
| SMOTE sur train uniquement | ❌ Sur tout | ✅ Post-split uniquement |
| Modularité | ❌ 1 fonction | ✅ 6 fonctions testables |
| Docstrings avec justifications | ❌ Absentes | ✅ Présentes |

**Enseignements clés :**

1. **La spécificité technique prime** — Mentionner explicitement les contraintes d'ordre (`optimize_memory` APRÈS `impute`) a évité un bug subtil qui aurait dégradé la précision silencieusement.

2. **Demander le "pourquoi" dans les docstrings** — En exigeant que chaque fonction documente sa justification, le code généré est devenu pédagogique et auditables par l'équipe.

3. **Décomposer en sous-tâches** — Un prompt demandant une seule grande fonction produit du code monolithique difficile à tester. Spécifier la décomposition souhaitée (`une fonction par étape`) a directement orienté l'architecture.

4. **Les contraintes négatives sont utiles** — Dire explicitement ce qu'il NE faut PAS faire (SMOTE sur tout le dataset, KNN sur données brutes) a permis d'éviter les erreurs classiques du preprocessing ML.

---

### Amélioration potentielle

Pour une prochaine itération, nous ajouterions dans le prompt : *"Génère également les tests unitaires pytest correspondants, en vérifiant les invariants de chaque fonction (shape, absence de NaN, types)"* — ce qui aurait produit les tests et le code en une seule itération.

---

## ⚠️ Avertissement

Ce projet est un **prototype académique** développé dans le cadre de la Coding Week. Il n'est **pas certifié** pour un usage clinique réel. Toute décision médicale doit reposer sur l'évaluation d'un professionnel de santé qualifié.

---

## 📚 Source des Données

Marek Sikora, Lukasz Wrobel — *Institute of Computer Science, Silesian University of Technology, Gliwice, Poland*

Dataset : [UCI ML Repository — Bone Marrow Transplant: Children](https://archive.ics.uci.edu/ml/datasets/Bone+Marrow+Transplant%3A+Children)
