# %% --- 1. INSTALLATION ET IMPORTS ---
# !pip install tabpfn --quiet

import pandas as pd
import numpy as np
import torch
import warnings
import gc
import optuna
from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# --- CONFIGURATION GLOBALE ---
SEED = 42
N_FOLDS = 5
OPTUNA_TIME_BUDGET = 1800  # 30 minutes de recherche (en secondes)
USE_GPU = torch.cuda.is_available()

# %% --- 2. FONCTIONS DE PRÉPARATION ---

def engineer_features(df):
    """Ingénierie de variables spécifique aux PME"""
    df = df.copy()
    df['business_expenses'] = df['business_expenses'].replace(0, 1)
    
    # Ratios et marges
    df['rev_per_expense'] = df['business_turnover'] / df['business_expenses']
    df['profit_margin'] = (df['business_turnover'] - df['business_expenses']) / df['business_turnover']
    df['income_efficiency'] = df['personal_income'] / df['business_expenses']
    
    # Log transform pour les colonnes monétaires
    monetary_cols = ['personal_income', 'business_expenses', 'business_turnover']
    for col in monetary_cols:
        df[f'log_{col}'] = np.log1p(df[col])
    
    # Agrégation des attitudes psychométriques
    att_cols = [c for c in df.columns if 'attitude' in c.lower() or 'perception' in c.lower()]
    if att_cols:
        df['att_mean'] = df[att_cols].mean(axis=1)
        df['att_std'] = df[att_cols].std(axis=1)
        
    return df

# %% --- 3. OPTIMISATION OPTUNA (CATBOOST) ---

def tune_catboost(X, y, cat_features, weights):
    def objective(trial):
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "task_type": "GPU" if USE_GPU else "CPU",
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1",
            "class_weights": weights,
            "random_seed": SEED,
            "verbose": 0,
            "allow_writing_files": False
        }
        
        # Ajout du device pour GPU si nécessaire (CatBoost gère souvent tout seul avec task_type='GPU', mais 'devices' aide parfois)
        if USE_GPU:
             params['devices'] = '0'

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        cv_scores = []
        
        for tr_idx, vl_idx in skf.split(X, y):
            X_tr, X_vl = X.iloc[tr_idx], X.iloc[vl_idx]
            y_tr, y_vl = y[tr_idx], y[vl_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), cat_features=cat_features, early_stopping_rounds=50)
            
            preds = model.predict(X_vl)
            cv_scores.append(f1_score(y_vl, preds, average='weighted'))
            
        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=OPTUNA_TIME_BUDGET)
    print(f"Meilleur F1 CatBoost : {study.best_value:.4f}")
    return study.best_params

# %% --- 4. PIPELINE DE STACKING ---

def run_grandmaster_pipeline(train_path, test_path):
    # 1. Chargement et Preprocessing
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    le = LabelEncoder()
    y = le.fit_transform(train['Target'])
    
    train = engineer_features(train)
    test = engineer_features(test)
    
    X = train.drop(columns=['ID', 'Target'])
    X_test = test.drop(columns=['ID'])
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    # Poids pour gérer le déséquilibre
    weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict = dict(zip(np.unique(y), weights))

    # 2. Tuning Optuna
    print("\n--- Phase 1 : Optimisation CatBoost ---")
    best_cat_params = tune_catboost(X, y, cat_features, weights)

    # 3. Training Cross-Validation
    print("\n--- Phase 2 : Entraînement OOF Stacking ---")
    oof_cat = np.zeros((len(X), 3)) 
    oof_xgb = np.zeros((len(X), 3))
    oof_pfn = np.zeros((len(X), 3))
    
    pred_cat = np.zeros((len(X_test), 3))
    pred_xgb = np.zeros((len(X_test), 3))
    pred_pfn = np.zeros((len(X_test), 3))

    # Pré-encodage pour XGB
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_enc = X.copy()
    X_test_enc = X_test.copy()
    X_enc[cat_features] = oe.fit_transform(X[cat_features].astype(str))
    X_test_enc[cat_features] = oe.transform(X_test[cat_features].astype(str))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y)):
        print(f"FOLD {fold+1}/{N_FOLDS}")
        X_tr, X_vl = X.iloc[tr_idx], X.iloc[vl_idx]
        X_tr_enc, X_vl_enc = X_enc.iloc[tr_idx], X_enc.iloc[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]

        # --- CATBOOST (Optimisé) ---
        cb_params = {**best_cat_params, "iterations": 2000, "task_type": "GPU" if USE_GPU else "CPU", 
                     "class_weights": weights, "random_seed": SEED, "verbose": 0}
        if USE_GPU:
             cb_params['devices'] = '0'
             
        m_cat = CatBoostClassifier(**cb_params)
        m_cat.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), cat_features=cat_features, early_stopping_rounds=100)
        oof_cat[vl_idx] = m_cat.predict_proba(X_vl)
        pred_cat += m_cat.predict_proba(X_test) / N_FOLDS

        # --- XGBOOST ---
        xgb_params = {
            'objective': 'multi:softprob', 
            'num_class': 3, 
            'n_estimators': 1000, 
            'learning_rate': 0.05, 
            'max_depth': 6
        }
        
        if USE_GPU:
            xgb_params.update({'device': 'cuda', 'tree_method': 'hist'})
        else:
            xgb_params.update({'tree_method': 'hist'})

        m_xgb = XGBClassifier(**xgb_params)
        m_xgb.fit(X_tr_enc, y_tr, sample_weight=pd.Series(y_tr).map(class_weights_dict), 
                  eval_set=[(X_vl_enc, y_vl)], verbose=False, early_stopping_rounds=50)
        
        oof_xgb[vl_idx] = m_xgb.predict_proba(X_vl_enc)
        pred_xgb += m_xgb.predict_proba(X_test_enc) / N_FOLDS

        # --- TabPFN (v2.5) ---
        # TabPFN v2.5 gère nativement de plus gros datasets et l'optimisation GPU
        m_pfn = TabPFNClassifier(device='cuda' if USE_GPU else 'cpu', random_state=SEED)
        
        # On utilise tout le dataset d'entraînement maintenant
        m_pfn.fit(X_tr, y_tr)
        oof_pfn[vl_idx] = m_pfn.predict_proba(X_vl)
        pred_pfn += m_pfn.predict_proba(X_test) / N_FOLDS

    # 4. MÉTA-MODÈLE (Stacking)
    X_stack_train = np.hstack([oof_cat, oof_xgb, oof_pfn])
    X_stack_test = np.hstack([pred_cat, pred_xgb, pred_pfn])
    
    meta_model = LogisticRegression(class_weight='balanced')
    meta_model.fit(X_stack_train, y)
    
    final_preds = meta_model.predict(X_stack_test)
    
    # 5. Export
    submission = pd.DataFrame({"ID": test["ID"], "Target": le.inverse_transform(final_preds)})
    submission.to_csv("submission_optuna_stacking.csv", index=False)
    print("\n✅ Terminé : submission_optuna_stacking.csv générée.")

if __name__ == "__main__":
    run_grandmaster_pipeline('Train.csv', 'Test.csv')
