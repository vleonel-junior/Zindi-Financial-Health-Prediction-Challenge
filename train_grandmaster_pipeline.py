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

# --- CONFIGURATION ---
SEED = 42
N_FOLDS = 5
MAX_TRIALS = 50
MAX_TIME = 3600  # 1 heure max par modèle
USE_GPU = torch.cuda.is_available()

# %% --- 2. FONCTIONS DE PRÉPARATION ---

def engineer_features(df):
    df = df.copy()
    df['business_expenses'] = df['business_expenses'].replace(0, 1)
    df['rev_per_expense'] = df['business_turnover'] / df['business_expenses']
    df['profit_margin'] = (df['business_turnover'] - df['business_expenses']) / df['business_turnover']
    for col in ['personal_income', 'business_expenses', 'business_turnover']:
        df[f'log_{col}'] = np.log1p(df[col])
    
    att_cols = [c for c in df.columns if 'attitude' in c.lower() or 'perception' in c.lower()]
    if att_cols:
        likert_map = {'strongly disagree': 1, 'disagree': 2, 'neither agree nor disagree': 3, 
                      'neutral': 3, 'agree': 4, 'strongly agree': 5, 'nan': 3}
        temp_att = pd.DataFrame()
        for col in att_cols:
            temp_att[col] = df[col].astype(str).str.lower().str.strip().map(likert_map).fillna(3)
        df['att_mean'] = temp_att.mean(axis=1)
        df['att_std'] = temp_att.std(axis=1)
    return df

def clean_categorical_nan(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].astype(str).replace(['nan', 'NaN', 'None'], 'missing')
    return df

# %% --- 3. TUNING DES MODÈLES ---

def tune_models(X, y, X_enc, cat_features, weights, cw_dict):
    # --- Tuning CatBoost ---
    def cb_objective(trial):
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 15),
            "task_type": "GPU" if USE_GPU else "CPU",
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1",
            "class_weights": list(weights),
            "random_seed": SEED, "verbose": 0
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for tr_idx, vl_idx in skf.split(X, y):
            m = CatBoostClassifier(**params)
            m.fit(X.iloc[tr_idx], y[tr_idx], eval_set=(X.iloc[vl_idx], y[vl_idx]), 
                  cat_features=cat_features, early_stopping_rounds=50)
            scores.append(f1_score(y[vl_idx], m.predict(X.iloc[vl_idx]), average='weighted'))
        return np.mean(scores)

    # --- Tuning XGBoost ---
    def xgb_objective(trial):
        params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "tree_method": 'gpu_hist' if USE_GPU else 'hist',
            "objective": 'multi:softprob', "num_class": 3, "random_state": SEED
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for tr_idx, vl_idx in skf.split(X_enc, y):
            m = XGBClassifier(**params)
            m.fit(X_enc.iloc[tr_idx], y[tr_idx], sample_weight=pd.Series(y[tr_idx]).map(cw_dict))
            scores.append(f1_score(y[vl_idx], m.predict(X_enc.iloc[vl_idx]), average='weighted'))
        return np.mean(scores)

    print("--- Tuning CatBoost (50 trials / 1h) ---")
    cb_study = optuna.create_study(direction="maximize")
    cb_study.optimize(cb_objective, n_trials=MAX_TRIALS, timeout=MAX_TIME)

    print("--- Tuning XGBoost (50 trials / 1h) ---")
    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(xgb_objective, n_trials=MAX_TRIALS, timeout=MAX_TIME)

    return cb_study.best_params, xgb_study.best_params

# %% --- 4. PIPELINE PRINCIPALE ---

def run_pipeline(train_path, test_path):
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    le = LabelEncoder()
    y = le.fit_transform(train['Target'])
    
    train, test = engineer_features(train), engineer_features(test)
    X, X_test = train.drop(columns=['ID', 'Target']), test.drop(columns=['ID'])
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X, X_test = clean_categorical_nan(X, cat_features), clean_categorical_nan(X_test, cat_features)

    weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    cw_dict = dict(zip(np.unique(y), weights))

    # Encoding pour Tuning XGB
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_enc = X.copy()
    X_enc[cat_features] = oe.fit_transform(X[cat_features])

    # PHASE TUNING
    best_cb, best_xgb = tune_models(X, y, X_enc, cat_features, weights, cw_dict)

    # PHASE STACKING
    print("--- Entraînement Final Stacking ---")
    X_test_enc = X_test.copy()
    X_test_enc[cat_features] = oe.transform(X_test[cat_features])
    
    oof_cat, oof_xgb, oof_pfn = [np.zeros((len(X), 3)) for _ in range(3)]
    pred_cat, pred_xgb, pred_pfn = [np.zeros((len(X_test), 3)) for _ in range(3)]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for tr_idx, vl_idx in skf.split(X, y):
        # CatBoost
        m_cb = CatBoostClassifier(**best_cb, iterations=2000, class_weights=list(weights), 
                                  task_type="GPU" if USE_GPU else "CPU", verbose=0)
        m_cb.fit(X.iloc[tr_idx], y[tr_idx], eval_set=(X.iloc[vl_idx], y[vl_idx]), cat_features=cat_features)
        oof_cat[vl_idx] = m_cb.predict_proba(X.iloc[vl_idx])
        pred_cat += m_cb.predict_proba(X_test) / N_FOLDS

        # XGBoost
        m_xgb = XGBClassifier(**best_xgb, n_estimators=2000)
        m_xgb.fit(X_enc.iloc[tr_idx], y[tr_idx], sample_weight=pd.Series(y[tr_idx]).map(cw_dict))
        oof_xgb[vl_idx] = m_xgb.predict_proba(X_enc.iloc[vl_idx])
        pred_xgb += m_xgb.predict_proba(X_test_enc) / N_FOLDS

        # TabPFN
        m_pfn = TabPFNClassifier(device='cuda' if USE_GPU else 'cpu', N_ensemble_configurations=16)
        m_pfn.fit(X.iloc[tr_idx[:1500]], y[tr_idx[:1500]])
        oof_pfn[vl_idx] = m_pfn.predict_proba(X.iloc[vl_idx])
        pred_pfn += m_pfn.predict_proba(X_test) / N_FOLDS

    # META-LEARNER
    X_stack_tr = np.hstack([oof_cat, oof_xgb, oof_pfn])
    X_stack_ts = np.hstack([pred_cat, pred_xgb, pred_pfn])
    meta_m = LogisticRegression(class_weight='balanced').fit(X_stack_tr, y)
    
    pd.DataFrame({"ID": test["ID"], "Target": le.inverse_transform(meta_m.predict(X_stack_ts))}).to_csv("submission_tuned.csv", index=False)
    print("🚀 Fichier submission_tuned.csv prêt !")

if __name__ == "__main__":
    run_pipeline('Train.csv', 'Test.csv')