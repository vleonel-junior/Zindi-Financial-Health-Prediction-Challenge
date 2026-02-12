
#%%
# Prétraitement et compréhension des données pour le challenge Zindi
import pandas as pd
import numpy as np

#%%
# 1. Lecture des fichiers de données
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')
var_defs = pd.read_csv('VariableDefinitions.csv')

#%%
# 2. Exploration rapide des fichiers
print("\n--- Aperçu du jeu d'entraînement ---")
print(train_df.head())

print("\n--- Aperçu du jeu de test ---")
print(test_df.head())

print("\n--- Définitions des variables ---")
print(var_defs.head())

#%%
# 3. Compréhension des variables et de la cible
print("\n--- Colonnes du jeu d'entraînement ---")
print(train_df.columns.tolist())

print("\n--- Colonnes du jeu de test ---")
print(test_df.columns.tolist())

#%%
# 4. Affichage de la description des variables (si disponible)
print("\n--- Aperçu des définitions de variables ---")
print(var_defs)

#%%
# 5. Types des variables
print("\n--- Types des colonnes ---")
print(train_df.dtypes)

categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
print("\n--- Variables catégorielles identifiées ---")
print(categorical_cols)

#%%
# 6. Inspection des modalités (Focus sur les variables catégorielles)
def show_modalities(df, limit=10):
    """
    Affiche les modalités (valeurs uniques) pour chaque colonne d'un DataFrame.

    Si le nombre de modalités dépasse 'limit', affiche seulement les premières et le total.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame à inspecter.
    limit : int, optional
        Le nombre maximum de modalités à afficher par colonne.
        Si None, affiche toutes les modalités. Par défaut 10.

    Returns
    -------
    None
    """
    if not isinstance(df, pd.DataFrame):
        print("Erreur : L'argument 'df' doit être un DataFrame pandas.")
        return

    print(f"\n--- Modalités des variables ({len(df.columns)} colonnes) ---")
    for col in df.columns:
        try:
            unique_vals = df[col].unique()
            num_unique = len(unique_vals)
            if limit is not None and num_unique > limit:
                print(f"{col=} ({num_unique} modalités) : {unique_vals[:limit]} ...")
            else:
                print(f"{col} ({num_unique} modalités) : {unique_vals}")
        except Exception as e:
            print(f"Erreur lors de l'inspection de la colonne '{col}': {e}")

# Appel de la fonction sur les variables catégorielles
show_modalities(train_df[categorical_cols], limit=None)

#%% 7. Nettoyage des chaînes de caractères (Apostrophes, Espaces, Encodage)

def clean_string_values(df):
    # On récupère les colonnes de type 'object' (catégorielles)
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        if col == 'ID': continue # On ne touche pas à l'ID
        
        # 1. Suppression des espaces en début/fin et passage en minuscule pour uniformiser
        df[col] = df[col].astype(str).str.strip()
        
        # 2. Correction des apostrophes et caractères d'encodage bizarres
        # On remplace les variantes de "Don't" et les caractères spéciaux
        df[col] = df[col].str.replace('’', "'").str.replace('‘', "'")
        df[col] = df[col].str.replace('?', 'o') # Pour corriger 'Don?t' -> 'Dont'
        df[col] = df[col].str.replace('\u200e', '') # Supprime les caractères invisibles LRM
        
        # 3. Harmonisation sémantique (Regroupement des synonymes)
        mapping_clean = {
            "Dont know or N/A": "Unknown",
            "Dont know": "Unknown",
            "Dont Know": "Unknown",
            "Do not know / N/A": "Unknown",
            "Dont know / doesnt apply": "Unknown",
            "Dont know (Do not show)": "Unknown",
            "Used to have but dont have now": "Used to have",
            "nan": "Missing", # On uniformise les valeurs nulles en texte pour l'instant
            "0": "No" # Correction spécifique pour cash_flow si '0' signifie 'No'
        }
        df[col] = df[col].replace(mapping_clean)
        
    return df

# Application du nettoyage sur Train et Test
train_df = clean_string_values(train_df)
test_df = clean_string_values(test_df)

#%% 8. Gestion des modalités manquantes ou exclusives (Alignement)

def align_categories(train, test, threshold=0.01):
    """
    Assure que le Test n'a pas de modalités inconnues au Train et 
    regroupe les catégories très rares.
    """
    cols_to_align = train.select_dtypes(include=['object']).columns.tolist()
    if 'Target' in cols_to_align: cols_to_align.remove('Target')
    if 'ID' in cols_to_align: cols_to_align.remove('ID')

    for col in cols_to_align:
        # 1. Calculer les fréquences sur le Train
        counts = train[col].value_counts(normalize=True)
        
        # 2. Identifier les modalités "Sûres" (Fréquence > 1% et présentes dans Train)
        safe_categories = counts[counts >= threshold].index.tolist()
        
        # 3. Appliquer la transformation : si pas dans 'safe_categories' -> 'Other'
        train[col] = train[col].apply(lambda x: x if x in safe_categories else 'Other')
        test[col] = test[col].apply(lambda x: x if x in safe_categories else 'Other')
        
        print(f"Variable {col} alignée. Modalités gardées : {len(safe_categories)}")

    return train, test

train_df, test_df = align_categories(train_df, test_df)

#%% 9. Vérification de l'alignement
print("\nVérification : Y a-t-il des modalités dans Test qui ne sont pas dans Train ?")
for col in train_df.select_dtypes(include=['object']).columns:
    if col != 'Target' and col != 'ID':
        diff = set(test_df[col]) - set(train_df[col])
        if diff:
            print(f"ALERTE sur {col}: Modalités orphelines dans Test: {diff}")
        else:
            print(f"{col}: OK (Parfaitement aligné)")

#%% 10. Imputation des variables numériques par Pays
num_cols = ['owner_age', 'personal_income', 'business_expenses', 
            'business_turnover', 'business_age_years', 'business_age_months']

def impute_numeric_by_country(df):
    for col in num_cols:
        # On remplace par la médiane du pays correspondant
        df[col] = df.groupby('country')[col].transform(lambda x: x.fillna(x.median()))
    return df

train_df = impute_numeric_by_country(train_df)
test_df = impute_numeric_by_country(test_df)

#%% 11. Feature Engineering : Création de nouvelles variables
def create_features(df):
    # a. Âge total de l'entreprise en mois
    df['total_business_age_months'] = (df['business_age_years'] * 12) + df['business_age_months']
    
    # b. Ratio de rentabilité théorique (Turnover / Expenses)
    # On ajoute 1 pour éviter la division par zéro
    df['expense_turnover_ratio'] = df['business_expenses'] / (df['business_turnover'] + 1)
    
    # c. Score de possession d'outils digitaux (Simple addition de variables binaires)
    # On transforme Yes/No en 1/0 temporairement pour le calcul
    digital_cols = ['has_mobile_money', 'has_cellphone', 'has_internet_banking', 'has_debit_card']
    df['digital_score'] = 0
    for col in digital_cols:
        df['digital_score'] += df[col].apply(lambda x: 1 if x == 'Have now' or x == 'Yes' else 0)
        
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)

#%% 12. Log-transformation pour réduire l'asymétrie
# Les revenus financiers ont souvent de très grands écarts (Power Law)
for col in ['personal_income', 'business_expenses', 'business_turnover']:
    train_df[col] = np.log1p(train_df[col]) # log1p gère le log(0)
    test_df[col] = np.log1p(test_df[col])

print("\n--- Nouvelles variables créées ---")
print(train_df[['total_business_age_months', 'expense_turnover_ratio', 'digital_score']].head())

#%% 13. Nettoyage final des NaNs (Correction du "Mean of empty slice")
# On utilise la médiane globale du Train comme "roue de secours" si un pays n'avait aucune donnée
train_medians = train_df.median(numeric_only=True)
train_df = train_df.fillna(train_medians)
test_df = test_df.fillna(train_medians)

#%% 14. Encodage de la Cible (Target) uniquement
# Obligatoire pour que le modèle sache quoi prédire (Low=0, Medium=1, High=2)
target_map = {'Low': 0, 'Medium': 1, 'High': 2}
if 'Target' in train_df.columns:
    train_df['Target'] = train_df['Target'].map(target_map)

#%% 15. Harmonisation des catégories (Le "Full Data" sans encodage)
# On s'assure que le type 'category' a exactement les mêmes modalités pour Train et Test.
# C'est crucial pour LightGBM et TabPFN.

cat_features = train_df.select_dtypes(include=['object']).columns.drop(['ID', 'Target'], errors='ignore')

for col in cat_features:
    # 1. On crée l'union des modalités présentes dans Train et Test
    full_data = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    
    # 2. On définit l'ordre et les modalités uniques sur l'ensemble complet
    unique_categories = full_data.unique()
    
    # 3. On transforme en type 'category' avec les mêmes catégories de référence
    train_df[col] = pd.Categorical(train_df[col], categories=unique_categories)
    test_df[col] = pd.Categorical(test_df[col], categories=unique_categories)

print(f"\n--- Alignement terminé sur {len(cat_features)} variables ---")

#%% 16. Sauvegarde des fichiers propres
# Note : Le format CSV ne stocke pas le type 'category'. 
# Il faudra juste faire un .astype('category') après le chargement pour CatBoost/LGBM.
train_df.to_csv('Train_Cleaned.csv', index=False)
test_df.to_csv('Test_Cleaned.csv', index=False)

print("\n--- Prétraitement terminé avec succès ! ---")
print(f"Forme finale du Train : {train_df.shape}")
print("Fichiers 'Train_Cleaned.csv' et 'Test_Cleaned.csv' créés.")

# Petit aperçu pour vérifier que c'est toujours du texte mais typé 'category'
print(train_df[cat_features].dtypes.head())
print(train_df['country'].head())

#%% 17. Évaluation et Préparation finale
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb # Crucial pour les callbacks
import gc

# On récupère la liste des colonnes catégorielles créée précédemment
cat_cols = train_df.select_dtypes(include=['category']).columns.tolist()

def evaluate_model(y_true, y_pred):
    score = f1_score(y_true, y_pred, average='weighted')
    print(f"F1 Score (Weighted): {score:.4f}")
    return score

# Préparation des matrices X et y
features = [col for col in train_df.columns if col not in ['ID', 'Target']]
X = train_df[features]
y = train_df['Target']
X_test = test_df[features]

#%% 23. PIPELINE GAGNANT - CROSS-VALIDATION PURE

from imblearn.over_sampling import SMOTENC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import gc

# Imports modèles
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
import torch
from tabpfn import TabPFNClassifier
from sklearn.preprocessing import OrdinalEncoder

# Encodeur pour TabPFN
tab_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
tab_encoder.fit(pd.concat([X[cat_cols], X_test[cat_cols]]))

# Poids de classes
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
weight_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}
print(f"\n>>> Poids de classes : {weight_dict}")

# Indices catégoriels pour SMOTE
cat_indices = [X.columns.get_loc(col) for col in cat_cols]

# ==========================================
# FONCTIONS D'ENTRAÎNEMENT
# ==========================================

def train_catboost_balanced(X_train, y_train, X_val, y_val, cat_features):
    sample_weights = y_train.map(weight_dict)
    
    model = CatBoostClassifier(
        iterations=2000,  # Augmenté pour laisser early stopping décider
        learning_rate=0.03,  # Réduit pour mieux généraliser
        depth=5,  # Réduit pour éviter overfitting
        loss_function='MultiClass',
        eval_metric='TotalF1',
        l2_leaf_reg=5,  # Régularisation importante !
        random_seed=42,
        verbose=False  # Moins de bruit
    )
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,  # Plus patient
        verbose=False
    )
    return model

def train_lgbm_balanced(X_train, y_train, X_val, y_val):
    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=20,  # Réduit (31 → 20)
        min_child_samples=30,  # Augmenté pour éviter overfitting
        subsample=0.8,  # Bagging
        colsample_bytree=0.8,  # Feature sampling
        reg_alpha=1.0,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        class_weight='balanced',
        objective='multiclass',
        metric='multi_logloss',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    return model

def train_tabpfn_balanced(X_train, y_train):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # SMOTE pour TabPFN
    smote = SMOTENC(
        categorical_features=cat_indices,
        sampling_strategy={2: int(len(y_train[y_train==2]) * 2.5)},
        random_state=42,
        k_neighbors=3
    )
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    model = TabPFNClassifier(device=device, N_ensemble_configurations=8)
    model.fit(X_train_sm, y_train_sm)
    return model

# ==========================================
# PIPELINE PRINCIPAL : CROSS-VALIDATION PURE
# ==========================================

def run_cv_pure(model_type, X, y, X_test, n_splits=5):
    """
    STRATÉGIE GAGNANTE :
    - Cross-Validation avec Out-of-Fold predictions
    - Moyenne des modèles de chaque fold sur le test (pas de full retrain)
    - SMOTE + Class Weights pour gérer le déséquilibre
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X), 3))
    test_preds = np.zeros((len(X_test), 3))
    
    # SMOTE configuré
    smote = SMOTENC(
        categorical_features=cat_indices,
        sampling_strategy={2: int(len(y[y==2]) * 2)},  # Doubler "High"
        random_state=42,
        k_neighbors=3
    )
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"📊 {model_type.upper()} - Fold {fold+1}/{n_splits}")
        print(f"{'='*60}")
        
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
        
        # SMOTE uniquement pour LGBM et CatBoost
        if model_type in ['lgbm', 'catboost']:
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
            print(f"SMOTE appliqué : {len(y_tr)} → {len(y_tr_sm)} samples")
        else:
            X_tr_sm, y_tr_sm = X_tr, y_tr
        
        # Entraînement
        if model_type == 'catboost':
            model = train_catboost_balanced(X_tr_sm, y_tr_sm, X_val, y_val, cat_cols)
            
        elif model_type == 'lgbm':
            model = train_lgbm_balanced(X_tr_sm, y_tr_sm, X_val, y_val)
            
        elif model_type == 'tabpfn':
            # Encoder pour TabPFN
            X_tr_tab = X_tr.copy(); X_val_tab = X_val.copy(); X_test_tab = X_test.copy()
            X_tr_tab[cat_cols] = tab_encoder.transform(X_tr_tab[cat_cols])
            X_val_tab[cat_cols] = tab_encoder.transform(X_val_tab[cat_cols])
            X_test_tab[cat_cols] = tab_encoder.transform(X_test_tab[cat_cols])
            X_tr_tab = X_tr_tab.astype(np.float32)
            X_val_tab = X_val_tab.astype(np.float32)
            X_test_tab = X_test_tab.astype(np.float32)
            
            model = train_tabpfn_balanced(X_tr_tab, y_tr)
        
        # Prédictions OOF
        if model_type == 'tabpfn':
            oof_preds[val_idx] = model.predict_proba(X_val_tab)
            test_preds += model.predict_proba(X_test_tab) / n_splits
        else:
            oof_preds[val_idx] = model.predict_proba(X_val)
            test_preds += model.predict_proba(X_test) / n_splits
        
        # Score du fold
        fold_score = f1_score(y_val, np.argmax(model.predict_proba(X_val if model_type != 'tabpfn' else X_val_tab), axis=1), average='weighted')
        fold_scores.append(fold_score)
        print(f"✅ Fold {fold+1} F1 Score : {fold_score:.4f}")
        
        gc.collect()
    
    # Score OOF global
    oof_score = f1_score(y, np.argmax(oof_preds, axis=1), average='weighted')
    print(f"\n{'='*60}")
    print(f"📈 {model_type.upper()} - Score OOF Final : {oof_score:.4f}")
    print(f"   Moyenne des folds : {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"{'='*60}")
    
    return oof_preds, test_preds

# ==========================================
# OPTIMISATION DES SEUILS
# ==========================================

def optimize_thresholds(y_true, y_probs):
    best_f1 = 0
    best_thresholds = None
    
    for t_high in np.arange(0.05, 0.40, 0.02):
        for t_medium in np.arange(0.20, 0.60, 0.05):
            preds = []
            for probs in y_probs:
                if probs[2] > t_high:
                    preds.append(2)
                elif probs[1] > t_medium:
                    preds.append(1)
                else:
                    preds.append(0)
            
            f1 = f1_score(y_true, preds, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = [t_medium, t_high]
    
    print(f"\n🎯 Seuils optimaux : Medium={best_thresholds[0]:.3f}, High={best_thresholds[1]:.3f}")
    print(f"   F1 optimisé : {best_f1:.4f}")
    return best_thresholds

def predict_with_thresholds(probs, thresholds):
    t_medium, t_high = thresholds
    preds = []
    for p in probs:
        if p[2] > t_high:
            preds.append(2)
        elif p[1] > t_medium:
            preds.append(1)
        else:
            preds.append(0)
    return np.array(preds)

# ==========================================
# EXÉCUTION
# ==========================================

print("\n" + "="*60)
print("🚀 ENTRAÎNEMENT - STRATÉGIE GAGNANTE")
print("="*60)

# Entraîner les 3 modèles
oof_cb, pred_cb = run_cv_pure('catboost', X, y, X_test)
oof_lgb, pred_lgb = run_cv_pure('lgbm', X, y, X_test)
oof_tab, pred_tab = run_cv_pure('tabpfn', X, y, X_test)

# Optimiser les seuils
print("\n" + "="*60)
print("🎯 OPTIMISATION DES SEUILS")
print("="*60)

thresh_cb = optimize_thresholds(y, oof_cb)
thresh_lgb = optimize_thresholds(y, oof_lgb)
thresh_tab = optimize_thresholds(y, oof_tab)

oof_ensemble = (oof_cb + oof_lgb + oof_tab) / 3
thresh_ensemble = optimize_thresholds(y, oof_ensemble)

# ==========================================
# SOUMISSIONS
# ==========================================

def save_submission_with_thresholds(probs, thresholds, name):
    preds_idx = predict_with_thresholds(probs, thresholds)
    labels = [inv_map[idx] for idx in preds_idx]
    
    dist = np.bincount(preds_idx, minlength=3) / len(preds_idx)
    print(f"\n{name} - Distribution : Low={dist[0]:.1%}, Med={dist[1]:.1%}, High={dist[2]:.1%}")
    
    sub = pd.DataFrame({'ID': test_df['ID'], 'Target': labels})
    filename = f'submission_{name}.csv'
    sub.to_csv(filename, index=False)
    print(f"✅ {filename}")

print("\n" + "="*60)
print("💾 GÉNÉRATION DES SOUMISSIONS")
print("="*60)

save_submission_with_thresholds(pred_cb, thresh_cb, "catboost_cv")
save_submission_with_thresholds(pred_lgb, thresh_lgb, "lgbm_cv")
save_submission_with_thresholds(pred_tab, thresh_tab, "tabpfn_cv")

final_test_probs = (pred_cb + pred_lgb + pred_tab) / 3
save_submission_with_thresholds(final_test_probs, thresh_ensemble, "ensemble_final")

print("\n🏆 Soumettez 'submission_ensemble_final.csv'")