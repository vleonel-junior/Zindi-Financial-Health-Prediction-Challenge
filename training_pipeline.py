# %% --- 1. IMPORTS ET CONFIGURATION ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import optuna 
from optuna.samplers import TPESampler

# Modèles
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Outils Scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Configuration
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# --- CONSTANTES ---
SEED = 42
N_FOLDS = 5
DO_OPTIMIZE = False  # Mettre à True pour lancer la recherche d'hyperparamètres (long)
USE_GPU = True       # ⚠️ Mettre à True si vous êtes sur Colab ou avez un GPU NVIDIA

# %% --- 2. FEATURE ENGINEERING ---

def process_data(df_train, df_test):
    """
    Traite train et test ensemble pour garantir la cohérence des features.
    """
    # Marqueur pour séparer plus tard
    df_train['is_train'] = 1
    df_test['is_train'] = 0
    df_test['Target'] = np.nan 
    
    # Concaténation
    full_df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    
    # --- A. NETTOYAGE TEXTE ---
    cat_cols = full_df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        full_df[col] = full_df[col].astype(str).str.lower().str.strip()

    # --- B. FEATURE ENGINEERING FINANCIER ---
    # Éviter la division par zéro
    full_df['business_expenses'] = full_df['business_expenses'].replace(0, 1)
    
    # Ratios financiers
    full_df['income_to_expense_ratio'] = full_df['personal_income'] / full_df['business_expenses']
    full_df['turnover_to_expense_ratio'] = full_df['business_turnover'] / full_df['business_expenses']
    full_df['margin_proxy'] = full_df['business_turnover'] - full_df['business_expenses']
    
    # Log transform (réduit l'impact des outliers comme les revenus très élevés)
    for col in ['personal_income', 'business_expenses', 'business_turnover']:
        full_df[f'log_{col}'] = np.log1p(full_df[col])

    # --- C. PSYCHOMÉTRIE (RISQUE/PRUDENCE) ---
    attitude_cols = [col for col in full_df.columns if 'attitude_' in col or 'perception_' in col]
    
    # On encode ces questions pour en faire des stats
    if len(attitude_cols) > 0:
        enc = OrdinalEncoder()
        # On utilise un DataFrame temporaire pour ne pas écraser les originales
        temp_attitude = enc.fit_transform(full_df[attitude_cols].astype(str))
        
        full_df['psychometric_score_sum'] = temp_attitude.sum(axis=1)
        full_df['psychometric_score_std'] = temp_attitude.std(axis=1)
        full_df['psychometric_zeros'] = (temp_attitude == 0).sum(axis=1) # Réponses "neutres/basses"

    # --- D. CONTEXTE GÉOGRAPHIQUE ---
    # Comparaison du CA par rapport à la moyenne du pays
    if 'country' in full_df.columns:
        map_mean = full_df.groupby('country')['business_turnover'].transform('mean')
        full_df['turnover_relative_to_country'] = full_df['business_turnover'] / map_mean

    return full_df

def get_class_weights_dict(y):
    """Calcule les poids pour équilibrer les classes (High/Low/Medium)."""
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    weight_dict = dict(zip(np.unique(y), weights))
    print(f"⚖️ Poids des classes calculés : {weight_dict}")
    return weight_dict

# %% --- 3. OPTIMISATION OPTUNA (AVEC SUPPORT GPU) ---

def objective_catboost(trial, X, y, cat_features, class_weights):
    """Fonction objectif pour Optuna (CatBoost)."""
    
    param = {
        'iterations': trial.suggest_int('iterations', 500, 1500),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        # Paramètres Fixes
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1',
        'class_weights': class_weights,
        'random_seed': SEED,
        'verbose': 0,
        'allow_writing_files': False
    }

    # --- ACTIVATION GPU ---
    if USE_GPU:
        param['task_type'] = 'GPU'
        param['devices'] = '0'
    # ----------------------

    # Cross-validation rapide (3 folds)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    f1_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**param, cat_features=cat_features)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)
        
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds, average='weighted'))

    return np.mean(f1_scores)

# %% --- 4. ENTRAÎNEMENT CROISÉ (CV) ---

def train_model_cv(X, y, X_test, model_type='catboost', class_weights_dict=None, best_params=None):
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # Arrays pour stocker les résultats
    oof_preds_proba = np.zeros((len(X), 3)) 
    test_preds_proba = np.zeros((len(X_test), 3))
    scores_f1 = []
    
    # Gestion des colonnes catégorielles
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Encodage LabelEncoder pour XGB et LGBM (CatBoost le fait tout seul)
    X_enc = X.copy()
    X_test_enc = X_test.copy()
    
    if model_type in ['xgb', 'lgbm']:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_enc[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))
        X_test_enc[cat_cols] = enc.transform(X_test[cat_cols].astype(str))

    print(f"\n🚀 Démarrage Entraînement : {model_type.upper()}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_enc, y)):
        X_train, y_train = X_enc.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X_enc.iloc[val_idx], y.iloc[val_idx]
        
        # --- CONFIGURATION DU MODÈLE ---
        model = None
        
        if model_type == 'catboost':
            params = best_params if best_params else {
                'iterations': 1500, 'learning_rate': 0.03, 'depth': 6,
                'l2_leaf_reg': 5,
                'loss_function': 'MultiClass', 'eval_metric': 'TotalF1',
                'class_weights': class_weights_dict, 
                'random_seed': SEED, 'allow_writing_files': False
            }
            # GPU Check
            if USE_GPU:
                params.update({'task_type': 'GPU', 'devices': '0'})

            model = CatBoostClassifier(**params, cat_features=cat_cols)
            model.fit(X.iloc[train_idx], y_train, eval_set=(X.iloc[val_idx], y_val), early_stopping_rounds=100, verbose=0)
            
            val_probs = model.predict_proba(X.iloc[val_idx])
            test_probs = model.predict_proba(X_test)

        elif model_type == 'lgbm':
            # Note: LGBM sur GPU demande souvent une compilation manuelle, on reste sur CPU
            params = best_params if best_params else {
                'n_estimators': 1500, 'learning_rate': 0.03, 'num_leaves': 31,
                'objective': 'multiclass', 'metric': 'multi_logloss',
                'class_weight': 'balanced',
                'random_state': SEED, 'verbose': -1, 'n_jobs': -1
            }
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            
            val_probs = model.predict_proba(X_val)
            test_probs = model.predict_proba(X_test_enc)

        elif model_type == 'xgb':
            # Pour XGB, on doit passer les poids échantillon par échantillon
            sample_weights = y_train.map(class_weights_dict)
            
            params = best_params if best_params else {
                'n_estimators': 1500, 'learning_rate': 0.03, 'max_depth': 6,
                'objective': 'multi:softprob', 'num_class': 3,
                'enable_categorical': False, 'random_state': SEED, 'n_jobs': -1
            }
            
            # GPU Check
            if USE_GPU:
                params.update({'tree_method': 'hist', 'device': 'cuda'})

            model = XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights, 
                      eval_set=[(X_val, y_val)], verbose=False)
            
            val_probs = model.predict_proba(X_val)
            test_probs = model.predict_proba(X_test_enc)

        # Stockage OOF
        oof_preds_proba[val_idx] = val_probs
        
        # Moyenne des prédictions Test
        test_preds_proba += test_probs / N_FOLDS
        
        # Calcul Score du Fold
        y_pred_lbl = np.argmax(val_probs, axis=1)
        f1 = f1_score(y_val, y_pred_lbl, average='weighted')
        scores_f1.append(f1)
        print(f"  > Fold {fold+1} F1 Weighted: {f1:.4f}")

    mean_f1 = np.mean(scores_f1)
    print(f"🏁 Moyenne F1 {model_type.upper()}: {mean_f1:.4f}")
    
    return oof_preds_proba, test_preds_proba, mean_f1

# %% --- 5. MAIN EXECUTION ---

if __name__ == "__main__":
    
    print(f"--- Configuration ---")
    print(f"GPU Activé: {USE_GPU}")
    print(f"Optimisation Optuna: {DO_OPTIMIZE}")

    # 1. Chargement
    print("\n--- Chargement des données ---")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    
    # 2. Encodage Cible
    le = LabelEncoder()
    train['Target_Encoded'] = le.fit_transform(train['Target'])
    y = train['Target_Encoded']
    
    # Mapping pour remettre les labels à la fin
    target_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    print(f"Mapping Target: {target_mapping}")
    
    # 3. Traitement
    print("--- Feature Engineering ---")
    full_df = process_data(train.drop(columns=['Target', 'Target_Encoded']), test)
    
    # Séparation X / X_test
    X = full_df[full_df['is_train'] == 1].drop(columns=['is_train', 'ID', 'Target'])
    X_test = full_df[full_df['is_train'] == 0].drop(columns=['is_train', 'ID', 'Target'])
    
    # 4. Calcul des poids (CRUCIAL pour le déséquilibre)
    class_weights_dict = get_class_weights_dict(y)
    
    # 5. Optimisation (Si activé)
    best_params_cat = None
    if DO_OPTIMIZE:
        print("\n--- Optimisation Optuna (CatBoost) ---")
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
        study.optimize(lambda trial: objective_catboost(trial, X, y, 
                                                       X.select_dtypes(include=['object']).columns.tolist(),
                                                       class_weights_dict), 
                       n_trials=20) 
        best_params_cat = study.best_params
        
        # On doit remettre les paramètres fixes
        best_params_cat.update({
            'loss_function': 'MultiClass', 'eval_metric': 'TotalF1',
            'class_weights': class_weights_dict, 'random_seed': SEED, 
            'allow_writing_files': False
        })
        print(f"Meilleurs params CatBoost: {study.best_params}")

    # 6. Entraînement des modèles
    print("\n--- Entraînement des Modèles ---")
    
    # CatBoost
    oof_cat, pred_cat, f1_cat = train_model_cv(X, y, X_test, 'catboost', 
                                              class_weights_dict=class_weights_dict,
                                              best_params=best_params_cat)
    
    # LightGBM
    oof_lgbm, pred_lgbm, f1_lgbm = train_model_cv(X, y, X_test, 'lgbm')
    
    # XGBoost
    oof_xgb, pred_xgb, f1_xgb = train_model_cv(X, y, X_test, 'xgb', 
                                              class_weights_dict=class_weights_dict)
    
    # 7. Ensemble (Moyenne Pondérée selon la performance F1)
    print("\n--- Création de l'Ensemble ---")
    total_f1 = f1_cat + f1_lgbm + f1_xgb
    w_cat = f1_cat / total_f1
    w_lgbm = f1_lgbm / total_f1
    w_xgb = f1_xgb / total_f1
    
    print(f"Poids de l'ensemble -> Cat: {w_cat:.2f}, LGBM: {w_lgbm:.2f}, XGB: {w_xgb:.2f}")
    
    final_probs = (pred_cat * w_cat) + (pred_lgbm * w_lgbm) + (pred_xgb * w_xgb)
    
    # Conversion en classes
    final_preds_indices = np.argmax(final_probs, axis=1)
    final_preds_labels = le.inverse_transform(final_preds_indices)
    
    # 8. Soumission
    submission = pd.DataFrame({
        "ID": test["ID"],
        "Target": final_preds_labels
    })
    
    filename = 'submission_optimized_ensemble_gpu.csv'
    submission.to_csv(filename, index=False)
    print(f"\n✅ Terminé ! Fichier '{filename}' généré avec succès.")
    
    # Stats finales
    print("\nDistribution des prédictions :")
    print(submission['Target'].value_counts(normalize=True))