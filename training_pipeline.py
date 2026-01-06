# %% --- 1. IMPORTS ET CONFIGURATION ---
import pandas as pd
import numpy as np
import warnings
import gc
import optuna 
from tqdm.auto import tqdm

# Modèles
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Outils Scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

# Désactivation des warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONSTANTES ---
SEED = 42
N_FOLDS = 5
MAX_TRIALS = 50
MAX_TIME = 3600 
USE_GPU = True  

# %% --- 2. FONCTIONS UTILITAIRES ---

class tqdm_callback:
    """Barre de progression pour Optuna"""
    def __init__(self, n_trials):
        self.pbar = tqdm(total=n_trials, desc="Optimisation")
    def __call__(self, study, trial):
        self.pbar.update(1)
        self.pbar.set_postfix({"Best F1": f"{study.best_value:.4f}"})

def process_data(train, test):
    le = LabelEncoder()
    y = le.fit_transform(train['Target'])
    df = pd.concat([train.drop('Target', axis=1), test], axis=0).reset_index(drop=True)
    
    # --- Feature Engineering Amélioré (Inspiré de train_lgbm.py) ---
    
    # 1. Imputation Logique ("No" par défaut pour les colonnes boolean-like)
    logical_cols = [
        'has_debit_card', 'has_mobile_money', 'has_loan_account', 'has_insurance',
        'medical_insurance', 'funeral_insurance', 'motor_vehicle_insurance',
        'has_internet_banking', 'has_credit_card'
    ]
    for col in logical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("No")

    # 2. Remplissage des inconnus
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].fillna("Unknown")

    # 3. Features Financières Clés
    df['business_expenses'] = df['business_expenses'].replace(0, 1) # Éviter div par zéro
    df['income_to_expense'] = df['personal_income'] / df['business_expenses']
    
    if 'personal_income' in df.columns and 'owner_age' in df.columns:
        df['income_per_age'] = df['personal_income'] / (df['owner_age'].replace(0, 1))

    # 4. Financial Access Score
    yes_vals = ['Yes', 'Have now', 'have now']
    access_cols = ['has_loan_account', 'has_internet_banking', 'has_debit_card', 'has_mobile_money']
    valid_cols = [c for c in access_cols if c in df.columns]
    if valid_cols:
        df['financial_access_score'] = df[valid_cols].isin(yes_vals).sum(axis=1)

    # 5. Log Transform
    for col in ['personal_income', 'business_expenses', 'business_turnover']:
        df[f'log_{col}'] = np.log1p(df[col])

    # 6. Psychométrie (Likert Mapping)
    likert_map = {'strongly disagree': 1, 'disagree': 2, 'neutral': 3,
                  'neither agree nor disagree': 3, 'agree': 4, 'strongly agree': 5, 'nan': 3}
    att_cols = [c for c in df.columns if 'attitude' in c.lower() or 'perception' in c.lower()]
    if att_cols:
        for col in att_cols:
            df[col] = df[col].astype(str).str.lower().str.strip().map(likert_map).fillna(3)
        df['psych_mean'] = df[att_cols].mean(axis=1)

    # 7. Nettoyage Catégoriel Final
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype(str).replace(['nan', 'None'], 'missing')
        
    return df.iloc[:len(train)], df.iloc[len(train):], y, le, cat_cols

# %% --- 3. TUNING (AVEC NOMS DE PARAMÈTRES CORRIGÉS) ---

def tune_models(X, y, cat_features, class_weights):
    weights_dict = dict(zip(np.unique(y), class_weights))
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_num = X.copy()
    X_num[cat_features] = oe.fit_transform(X[cat_features].astype(str))

    # --- CATBOOST ---
    print("\n" + "="*50 + "\n🔥 ÉTAPE 1: OPTIMISATION CATBOOST\n" + "="*50)
    def obj_cb(trial):
        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'task_type': 'GPU' if USE_GPU else 'CPU',
            'class_weights': list(class_weights),
            'random_seed': SEED, 'verbose': 0
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for t, v in skf.split(X, y):
            m = CatBoostClassifier(**params)
            m.fit(X.iloc[t], y.iloc[t], cat_features=cat_features, early_stopping_rounds=50)
            scores.append(f1_score(y.iloc[v], m.predict(X.iloc[v]), average='weighted'))
        return np.mean(scores)

    study_cb = optuna.create_study(direction='maximize')
    study_cb.optimize(obj_cb, n_trials=MAX_TRIALS, timeout=MAX_TIME, callbacks=[tqdm_callback(MAX_TRIALS)])

    # --- XGBOOST ---
    print("\n" + "="*50 + "\n🚀 ÉTAPE 2: OPTIMISATION XGBOOST\n" + "="*50)
    def obj_xgb(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'tree_method': 'hist', 'device': 'cuda' if USE_GPU else 'cpu',
            'random_state': SEED
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for t, v in skf.split(X_num, y):
            m = XGBClassifier(**params)
            m.fit(X_num.iloc[t], y.iloc[t], sample_weight=y.iloc[t].map(weights_dict))
            scores.append(f1_score(y.iloc[v], m.predict(X_num.iloc[v]), average='weighted'))
        return np.mean(scores)

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(obj_xgb, n_trials=MAX_TRIALS, timeout=MAX_TIME, callbacks=[tqdm_callback(MAX_TRIALS)])

    # --- LIGHTGBM ---
    print("\n" + "="*50 + "\n💡 ÉTAPE 3: OPTIMISATION LIGHTGBM\n" + "="*50)
    def obj_lgbm(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'device': 'gpu' if USE_GPU else 'cpu',
            'class_weight': 'balanced', 'random_state': SEED, 'verbose': -1
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for t, v in skf.split(X_num, y):
            m = LGBMClassifier(**params)
            m.fit(X_num.iloc[t], y.iloc[t])
            scores.append(f1_score(y.iloc[v], m.predict(X_num.iloc[v]), average='weighted'))
        return np.mean(scores)

    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(obj_lgbm, n_trials=MAX_TRIALS, timeout=MAX_TIME, callbacks=[tqdm_callback(MAX_TRIALS)])

    return study_cb.best_params, study_xgb.best_params, study_lgbm.best_params

# %% --- 4. EXECUTION ET STACKING ---

if __name__ == "__main__":
    # Chargement
    train_raw, test_raw = pd.read_csv('Train.csv'), pd.read_csv('Test.csv')
    X, X_test, y, le, cat_cols = process_data(train_raw, test_raw)
    y_series = pd.Series(y)
    
    # Calcul des poids de classes
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weights_dict = dict(zip(np.unique(y), class_weights))

    # 1. Tuning
    bp_cb, bp_xgb, bp_lgbm = tune_models(X, y_series, cat_cols, class_weights)
    
    # 2. Entraînement Final (Cross-Validation)
    print("\n" + "="*50 + "\n🏗️ ÉTAPE 4: ENTRAÎNEMENT DE L'ENSEMBLE (5-FOLDS)\n" + "="*50)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_stack = np.zeros((len(X), 9)) # 3 modèles * 3 classes
    test_stack = np.zeros((len(X_test), 9))
    
    # Encodage numérique pour XGB/LGBM
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_num, X_test_num = X.copy(), X_test.copy()
    X_num[cat_cols] = oe.fit_transform(X[cat_cols].astype(str))
    X_test_num[cat_cols] = oe.transform(X_test[cat_cols].astype(str))

    for f, (t_idx, v_idx) in enumerate(tqdm(skf.split(X, y), total=N_FOLDS, desc="Folds CV")):
        # CatBoost
        m_cb = CatBoostClassifier(**bp_cb, iterations=2000, task_type='GPU' if USE_GPU else 'CPU', 
                                  class_weights=list(class_weights), verbose=0)
        m_cb.fit(X.iloc[t_idx], y_series.iloc[t_idx], cat_features=cat_cols)
        oof_stack[v_idx, 0:3] = m_cb.predict_proba(X.iloc[v_idx])
        test_stack[:, 0:3] += m_cb.predict_proba(X_test) / N_FOLDS
        
        # XGBoost
        m_xgb = XGBClassifier(**bp_xgb, n_estimators=2000, tree_method='hist', device='cuda' if USE_GPU else 'cpu')
        m_xgb.fit(X_num.iloc[t_idx], y_series.iloc[t_idx], sample_weight=y_series.iloc[t_idx].map(weights_dict))
        oof_stack[v_idx, 3:6] = m_xgb.predict_proba(X_num.iloc[v_idx])
        test_stack[:, 3:6] += m_xgb.predict_proba(X_test_num) / N_FOLDS
        
        # LightGBM
        m_lgbm = LGBMClassifier(**bp_lgbm, n_estimators=2000, class_weight='balanced')
        m_lgbm.fit(X_num.iloc[t_idx], y_series.iloc[t_idx])
        oof_stack[v_idx, 6:9] = m_lgbm.predict_proba(X_num.iloc[v_idx])
        test_stack[:, 6:9] += m_lgbm.predict_proba(X_test_num) / N_FOLDS

    # 3. Meta-Learner Final
    print("\n" + "="*50 + "\n🎯 ÉTAPE 5: MÉTA-MODÈLE & GÉNÉRATION CSV\n" + "="*50)
    meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    meta_model.fit(oof_stack, y)
    
    final_preds = meta_model.predict(test_stack)
    
    # Soumission
    submission = pd.DataFrame({
        "ID": test_raw["ID"],
        "Target": le.inverse_transform(final_preds)
    })
    submission.to_csv("submission_master_final.csv", index=False)
    
    print("✅ Succès ! Fichier 'submission_master_final.csv' généré.")
    print("\nDistribution des prédictions :")
    print(submission['Target'].value_counts(normalize=True))