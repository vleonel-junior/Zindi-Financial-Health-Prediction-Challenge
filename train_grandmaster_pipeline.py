# %% --- 1. INSTALLATION ET IMPORTS ---
import pandas as pd
import numpy as np
import torch
import warnings
import gc
import optuna
from tqdm.auto import tqdm

# Modèles
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

# On essaie d'importer TabPFN, sinon on l'ignore pour éviter les crashs
try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIGURATION ---
SEED = 42
N_FOLDS = 5
MAX_TRIALS = 50
MAX_TIME = 3600 
USE_GPU = torch.cuda.is_available()

# %% --- 2. FONCTIONS DE PRÉPARATION ---

class tqdm_callback:
    def __init__(self, n_trials, desc):
        self.pbar = tqdm(total=n_trials, desc=desc)
    def __call__(self, study, trial):
        self.pbar.update(1)
        self.pbar.set_postfix({"Best_F1": f"{study.best_value:.4f}"})

def engineer_features(df):
    df = df.copy()
    # Correction des divisions par zéro (Cause de l'erreur inf)
    df['business_expenses'] = df['business_expenses'].replace(0, 1)
    df['business_turnover_safe'] = df['business_turnover'].replace(0, 1)
    
    df['rev_per_expense'] = df['business_turnover'] / df['business_expenses']
    df['profit_margin'] = (df['business_turnover'] - df['business_expenses']) / df['business_turnover_safe']
    
    for col in ['personal_income', 'business_expenses', 'business_turnover']:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0)) # clip pour éviter log de négatif
    
    att_cols = [c for c in df.columns if 'attitude' in c.lower() or 'perception' in c.lower()]
    if att_cols:
        likert_map = {'strongly disagree': 1, 'disagree': 2, 'neither agree nor disagree': 3, 
                      'neutral': 3, 'agree': 4, 'strongly agree': 5, 'nan': 3}
        temp_att = pd.DataFrame()
        for col in att_cols:
            temp_att[col] = df[col].astype(str).str.lower().str.strip().map(likert_map).fillna(3)
        df['att_mean'] = temp_att.mean(axis=1)
        df['att_std'] = temp_att.std(axis=1)
    
    return df.drop(columns=['business_turnover_safe'])

def clean_categorical_nan(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].astype(str).replace(['nan', 'NaN', 'None'], 'missing')
    return df

# %% --- 3. TUNING DES MODÈLES ---

def tune_models(X, y, X_enc, cat_features, weights, cw_dict):
    # --- Tuning CatBoost ---
    print("\n" + "="*50 + "\n🔥 ÉTAPE 1: OPTIMISATION CATBOOST\n" + "="*50)
    def cb_objective(trial):
        params = {
            "iterations": 800,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 15),
            "task_type": "GPU" if USE_GPU else "CPU",
            "class_weights": list(weights),
            "random_seed": SEED, "verbose": 0
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = [f1_score(y[vl], CatBoostClassifier(**params).fit(X.iloc[tr], y[tr], cat_features=cat_features, early_stopping_rounds=30).predict(X.iloc[vl]), average='weighted') 
                  for tr, vl in skf.split(X, y)]
        return np.mean(scores)

    cb_study = optuna.create_study(direction="maximize")
    cb_study.optimize(cb_objective, n_trials=MAX_TRIALS, timeout=MAX_TIME, callbacks=[tqdm_callback(MAX_TRIALS, "CatBoost")])

    # --- Tuning XGBoost ---
    print("\n" + "="*50 + "\n🚀 ÉTAPE 2: OPTIMISATION XGBOOST\n" + "="*50)
    def xgb_objective(trial):
        params = {
            "n_estimators": 800,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "tree_method": 'gpu_hist' if USE_GPU else 'hist',
            "random_state": SEED
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for tr, vl in skf.split(X_enc, y):
            m = XGBClassifier(**params)
            m.fit(X_enc.iloc[tr], y[tr], sample_weight=pd.Series(y[tr]).map(cw_dict))
            scores.append(f1_score(y[vl], m.predict(X_enc.iloc[vl]), average='weighted'))
        return np.mean(scores)

    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(xgb_objective, n_trials=MAX_TRIALS, timeout=MAX_TIME, callbacks=[tqdm_callback(MAX_TRIALS, "XGBoost")])

    return cb_study.best_params, xgb_study.best_params

# %% --- 4. PIPELINE PRINCIPALE ---

def run_pipeline(train_path, test_path):
    train_raw, test_raw = pd.read_csv(train_path), pd.read_csv(test_path)
    le = LabelEncoder()
    y = le.fit_transform(train_raw['Target'])
    
    train = engineer_features(train_raw)
    test = engineer_features(test_raw)
    
    X = train.drop(columns=['ID', 'Target'])
    X_test = test.drop(columns=['ID'])
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X = clean_categorical_nan(X, cat_features)
    X_test = clean_categorical_nan(X_test, cat_features)

    weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    cw_dict = dict(zip(np.unique(y), weights))

    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_enc = X.copy()
    X_enc[cat_features] = oe.fit_transform(X[cat_features])
    X_test_enc = X_test.copy()
    X_test_enc[cat_features] = oe.transform(X_test[cat_features])

    # TUNING
    best_cb, best_xgb = tune_models(X, y, X_enc, cat_features, weights, cw_dict)

    # STACKING
    print("\n" + "="*50 + "\n🏗️ ÉTAPE 3: ENTRAÎNEMENT FINAL (5-FOLDS)\n" + "="*50)
    
    oof_stack = np.zeros((len(X), 6 if not HAS_TABPFN else 9))
    pred_stack = np.zeros((len(X_test), 6 if not HAS_TABPFN else 9))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for f, (tr_idx, vl_idx) in enumerate(tqdm(skf.split(X, y), total=N_FOLDS, desc="Folds CV")):
        # 1. CatBoost
        m_cb = CatBoostClassifier(**best_cb, iterations=1500, task_type="GPU" if USE_GPU else "CPU", verbose=0)
        m_cb.fit(X.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
        oof_stack[vl_idx, 0:3] = m_cb.predict_proba(X.iloc[vl_idx])
        pred_stack[:, 0:3] += m_cb.predict_proba(X_test) / N_FOLDS

        # 2. XGBoost
        m_xgb = XGBClassifier(**best_xgb, n_estimators=1500)
        m_xgb.fit(X_enc.iloc[tr_idx], y[tr_idx], sample_weight=pd.Series(y[tr_idx]).map(cw_dict))
        oof_stack[vl_idx, 3:6] = m_xgb.predict_proba(X_enc.iloc[vl_idx])
        pred_stack[:, 3:6] += m_xgb.predict_proba(X_test_enc) / N_FOLDS

        # 3. TabPFN (Optionnel car très lent)
        if HAS_TABPFN:
            m_pfn = TabPFNClassifier(device='cuda' if USE_GPU else 'cpu', N_ensemble_configurations=16)
            # TabPFN est limité en taille de données, on prend un échantillon si trop gros
            idx_pfn = tr_idx[:2000] if len(tr_idx) > 2000 else tr_idx
            m_pfn.fit(X.iloc[idx_pfn], y[idx_pfn], overwrite_warning=True)
            oof_stack[vl_idx, 6:9] = m_pfn.predict_proba(X.iloc[vl_idx])
            pred_stack[:, 6:9] += m_pfn.predict_proba(X_test) / N_FOLDS

    # META-LEARNER
    print("\n" + "="*50 + "\n🎯 ÉTAPE 4: GÉNÉRATION SOUMISSION\n" + "="*50)
    meta_m = LogisticRegression(class_weight='balanced').fit(oof_stack, y)
    final_preds = meta_m.predict(pred_stack)
    
    sub = pd.DataFrame({"ID": test_raw["ID"], "Target": le.inverse_transform(final_preds)})
    sub.to_csv("submission_tuned_v2.csv", index=False)
    print("🚀 Terminé ! Fichier 'submission_tuned_v2.csv' généré.")

if __name__ == "__main__":
    run_pipeline('Train.csv', 'Test.csv')